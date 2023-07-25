# -*- coding: utf-8 -*-
import copy
import os
import time

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision
from pytorch_lightning.loggers import TensorBoardLogger

from lightly.data import (LightlyDataset, SimCLRCollateFunction, collate)
from lightly.loss import (NTXentLoss)
from lightly.models import ResNetGenerator, utils
from lightly.models.modules import heads
from lightly.utils.benchmarking import BenchmarkModule

logs_root_dir = os.path.join(os.getcwd(), "benchmark_logs")

# set max_epochs to 800 for long run (takes around 10h on a single V100)
max_epochs, num_workers, knn_k, knn_t, classes = 800, 8, 200, 0.1, 10

# Set to True to enable Synchronized Batch Norm (requires distributed=True).
# If enabled the batch norm is calculated over all gpus, otherwise the batch
# norm is only calculated from samples on the same gpu.
sync_batchnorm = False
n_runs = 1  # optional, increase to create multiple runs and report mean + std
batch_size = 512
lr_factor = 1.0

# use a GPU if available
# limit to single gpu if not using distributed training
gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
distributed_backend = None
gpus = min(gpus, 1)

path_to_train = "cifar10/cifar10/train/"
path_to_test = "cifar10/cifar10/test/"

# Use SimCLR augmentations, additionally, disable blur for cifar10
collate_fn = SimCLRCollateFunction(input_size=32, gaussian_blur=0.0)

# No additional augmentations for the test set
test_transforms = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=collate.imagenet_normalize["mean"],
            std=collate.imagenet_normalize["std"],
        ),
    ]
)

# we use test transformations for getting the feature for kNN on train data
dataset_train_ssl = LightlyDataset(input_dir=path_to_train)
dataset_train_kNN = LightlyDataset(input_dir=path_to_train, transform=test_transforms)
dataset_test = LightlyDataset(input_dir=path_to_test, transform=test_transforms)

def compute_gce(z1_outs, z2_outs, quantile):
      import math
      from scipy.sparse.csgraph import reverse_cuthill_mckee
      from scipy.sparse import csr_matrix
      import statistics
      # (1) Stack and normalize outputs
      src_train_full = torch.nn.functional.normalize(torch.vstack(z1_outs)).cuda()
      tgt_train_full = torch.nn.functional.normalize(torch.vstack(z2_outs)).cuda()
      z1_outs, z2_outs = [], []

      # (2) Estimate quantile
      chunk_size, num_samples, quantiles = 10, len(tgt_train_full), []
      for chunk_idx in range(math.ceil(len(tgt_train_full)/chunk_size)):
        mat_val = src_train_full[chunk_idx*chunk_size:(chunk_idx+1)*chunk_size] @ tgt_train_full.T
        quantiles.append(float(torch.quantile(mat_val, quantile)))

      # (3) Get similarity graph thresholded on quantile
      row, col, data, quantile = [], [], [], statistics.median(quantiles)
      for chunk_idx in range(math.ceil(len(src_train_full)/chunk_size)):
        mat_val = src_train_full[chunk_idx*chunk_size:(chunk_idx+1)*chunk_size] @ tgt_train_full.T
        ret = (mat_val.flatten() > quantile).nonzero(as_tuple=True)[0].cpu()
        row += ((ret - (ret % num_samples))/num_samples + chunk_idx*chunk_size).int().tolist()
        col += (ret % num_samples).tolist()
        data += [1.0 for _ in range(len(ret))]

      # (4) Get permutation using graph bandwidth minimization on sparsified graph (cuthill-mckee)
      permutation = list(reverse_cuthill_mckee(csr_matrix((data, (row, col)),
                                                    shape=(num_samples, num_samples))))
      return permutation

def get_data_loaders(batch_size: int, model):
    """Helper method to create dataloaders for ssl, kNN train and kNN test

    Args:
        batch_size: Desired batch size for all dataloaders
    """
    col_fn = collate_fn
    dataloader_train_ssl = torch.utils.data.DataLoader(
        dataset_train_ssl,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=col_fn,
        drop_last=False,
        num_workers=num_workers,
    )

    dataloader_train_kNN = torch.utils.data.DataLoader(
        dataset_train_kNN,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )

    dataloader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )

    return dataloader_train_ssl, dataloader_train_kNN, dataloader_test, dataset_train_ssl

class SimCLRModel(BenchmarkModule):
    def __init__(self, dataloader_kNN, num_classes):
        super().__init__(dataloader_kNN, num_classes)
        # create a ResNet backbone and remove the classification head
        resnet = ResNetGenerator("resnet-18")
        self.backbone = nn.Sequential(
            *list(resnet.children())[:-1], nn.AdaptiveAvgPool2d(1)
        )
        self.projection_head = heads.SimCLRProjectionHead(512, 512, 128)
        self.criterion = NTXentLoss()

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(x)
        return z

    def _prepare_epoch(self, trainer, model):
        trainer.datamodule.set_phase(model)

    def on_train_epoch_start(self):
        self._prepare_epoch(self.trainer, self)

    def training_step(self, batch, batch_index):
        (x0, x1), _, _ = batch
        z0 = self.forward(x0)
        z1 = self.forward(x1)
        loss = self.criterion(z0, z1)
        self.log("train_loss_ssl", loss)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.SGD(
            self.parameters(), lr=6e-2 * lr_factor, momentum=0.9, weight_decay=5e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, max_epochs)
        return [optim], [scheduler]

class ContrastiveDataModule(pl.LightningDataModule):
    def __init__(self, train_dataloader, train_dataset, valid_dataloader):
        super().__init__()
        self.train_dataloader_start = train_dataloader
        self.train_dataset_start = train_dataset
        self.train_dataloader_current = train_dataloader
        self.valid_dataloader = valid_dataloader

    def set_phase(self, model):
        with torch.no_grad():
            eval_z1, eval_z2 = [], []
            for batch in self.train_dataloader_start:
                #get code and nl vectors
                (x0, x1), labs, _ = batch
                z0 = model(x0.cuda())
                z1 = model(x1.cuda())

                eval_z1.append(z0.cpu().float())
                eval_z2.append(z1.cpu().float())

            permutation = compute_gce(eval_z1, eval_z2, 0.995)
            self.train_dataset = torch.utils.data.Subset(self.train_dataset_start, permutation)
            self.train_dataloader_current = torch.utils.data.DataLoader(self.train_dataset,
                                                                        batch_size=batch_size,
                                                                        shuffle=False,
                                                                        collate_fn=collate_fn,
                                                                        drop_last=False,
                                                                        num_workers=num_workers)

    def train_dataloader(self): return self.train_dataloader_current
    def val_dataloader(self): return self.valid_dataloader

class MocoModel(BenchmarkModule):
    def __init__(self, dataloader_kNN, num_classes):
        super().__init__(dataloader_kNN, num_classes)

        # create a ResNet backbone and remove the classification head
        num_splits = 0 if sync_batchnorm else 8
        resnet = ResNetGenerator("resnet-18", num_splits=num_splits)
        self.backbone = nn.Sequential(
            *list(resnet.children())[:-1], nn.AdaptiveAvgPool2d(1)
        )

        # create a moco model based on ResNet
        self.projection_head = heads.MoCoProjectionHead(512, 512, 128)
        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)
        utils.deactivate_requires_grad(self.backbone_momentum)
        utils.deactivate_requires_grad(self.projection_head_momentum)

        # create our loss with the optional memory bank
        self.criterion = NTXentLoss(
            temperature=0.1,
            memory_bank_size=8192,
        )

    def _prepare_epoch(self, trainer, model):
        trainer.datamodule.set_phase(model)

    def on_train_epoch_start(self):
        self._prepare_epoch(self.trainer, self)

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        return self.projection_head(x)

    def training_step(self, batch, batch_idx):
        (x0, x1), _, _ = batch

        # update momentum
        utils.update_momentum(self.backbone, self.backbone_momentum, 0.99)
        utils.update_momentum(self.projection_head, self.projection_head_momentum, 0.99)

        def step(x0_, x1_):
            x1_, shuffle = utils.batch_shuffle(x1_, distributed=False)
            x0_ = self.backbone(x0_).flatten(start_dim=1)
            x0_ = self.projection_head(x0_)

            x1_ = self.backbone_momentum(x1_).flatten(start_dim=1)
            x1_ = self.projection_head_momentum(x1_)
            x1_ = utils.batch_unshuffle(x1_, shuffle, distributed=False)
            return x0_, x1_

        # We use a symmetric loss (model trains faster at little compute overhead)
        # https://colab.research.google.com/github/facebookresearch/moco/blob/colab-notebook/colab/moco_cifar10_demo.ipynb
        loss_1 = self.criterion(*step(x0, x1))
        loss_2 = self.criterion(*step(x1, x0))

        loss = 0.5 * (loss_1 + loss_2)
        self.log("train_loss_ssl", loss)
        return loss

    def configure_optimizers(self):
        params = list(self.backbone.parameters()) + list(
            self.projection_head.parameters()
        )
        optim = torch.optim.SGD(
            params,
            lr=6e-2 * lr_factor,
            momentum=0.9,
            weight_decay=5e-4,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, max_epochs)
        return [optim], [scheduler]

models = [MocoModel]
bench_results = dict()
experiment_version = None
# loop through configurations and train models
for BenchmarkModel in models:
    runs = []
    model_name = BenchmarkModel.__name__.replace("Model", "")
    for seed in range(n_runs):
        pl.seed_everything(seed)
        dataloader_train_ssl, dataloader_train_kNN, dataloader_test, dataset_train_ssl = get_data_loaders(
            batch_size=batch_size,
            model=BenchmarkModel,
        )
        benchmark_model = BenchmarkModel(dataloader_train_kNN, classes)

        # Save logs to: {CWD}/benchmark_logs/cifar10/{experiment_version}/{model_name}/
        # If multiple runs are specified a subdirectory for each run is created.
        sub_dir = model_name if n_runs <= 1 else f"{model_name}/run{seed}"
        logger = TensorBoardLogger(
            save_dir=os.path.join(logs_root_dir, "cifar10"),
            name="",
            sub_dir=sub_dir,
            version=experiment_version,
        )
        if experiment_version is None:
            # Save results of all models under same version directory
            experiment_version = logger.version
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=os.path.join(logger.log_dir, "checkpoints")
        )
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            gpus=gpus,
            default_root_dir=logs_root_dir,
            strategy=distributed_backend,
            sync_batchnorm=sync_batchnorm,
            logger=logger,
            callbacks=[checkpoint_callback],
            reload_dataloaders_every_n_epochs = 1,
        )
        start = time.time()
        trainer.fit(
            benchmark_model,
            datamodule = ContrastiveDataModule(dataloader_train_ssl, dataset_train_ssl, dataloader_test),
        )
        end = time.time()
        run = {
            "model": model_name,
            "batch_size": batch_size,
            "epochs": max_epochs,
            "max_accuracy": benchmark_model.max_accuracy,
            "runtime": end - start,
            "gpu_memory_usage": torch.cuda.max_memory_allocated(),
            "seed": seed,
        }
        runs.append(run)
        print(run)

        # delete model and trainer + free up cuda memory
        del benchmark_model
        del trainer
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    bench_results[model_name] = runs

# print results table
header = (
    f"| {'Model':<13} | {'Batch Size':>10} | {'Epochs':>6} "
    f"| {'KNN Test Accuracy':>18} | {'Time':>10} | {'Peak GPU Usage':>14} |"
)
print("-" * len(header))
print(header)
print("-" * len(header))
for model, results in bench_results.items():
    runtime = np.array([result["runtime"] for result in results])
    runtime = runtime.mean() / 60  # convert to min
    accuracy = np.array([result["max_accuracy"] for result in results])
    gpu_memory_usage = np.array([result["gpu_memory_usage"] for result in results])
    gpu_memory_usage = gpu_memory_usage.max() / (1024**3)  # convert to gbyte

    if len(accuracy) > 1:
        accuracy_msg = f"{accuracy.mean():>8.3f} +- {accuracy.std():>4.3f}"
    else:
        accuracy_msg = f"{accuracy.mean():>18.3f}"

    print(
        f"| {model:<13} | {batch_size:>10} | {max_epochs:>6} "
        f"| {accuracy_msg} | {runtime:>6.1f} Min "
        f"| {gpu_memory_usage:>8.1f} GByte |",
        flush=True,
    )
print("-" * len(header))