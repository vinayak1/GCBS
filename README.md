# Global Contrastive Batch Sampling
Repository for the ICML 2023 paper "Global Selection of Contrastive Batches via Optimization on Sample Permutations" [https://proceedings.mlr.press/v202/sachidananda23a/sachidananda23a.pdf]

## Reproduction of SOTA Paper Results

Experimentation code for the code search and sentence embedding experiments is provided in three folders. Each folder contains an iPython notebook with instructions for running the code.
- `GCBS_UniXcoder/`: Code search experiments (CosQA, AdvTest, CodeSearchNet), GCBS provides a 1.0, 2.0 and 2.2 percent improvement on state-of-the-art models.
- `GCBS_SimCSE/`: Sentence Embedding experiments (STS) with the SimCSE model, GCBS provides a 1.03 percent improvement on state-of-the-art models.
- `GCBS_PromCSE/`: Sentence Embedding experiments (STS) with the PromCSE model, GCBS provides a 0.27 percent improvement on state-of-the-art models.

All reported results were run on an individual Nvidia A100 GPU with the following CUDA and torch versions:
```
CUDA Version: 11.6
torch==1.11.0
```

TODO: Provide hyperparameters and docker image for replication.

## Self-supervised Image Classification experiments

CIFAR10 experimentation ported from [https://docs.lightly.ai/self-supervised-learning/getting_started/benchmarks.html] is provided in `cifar_moco_pytorch_lightning_gcbs.py`. This code uses PyTorch Lightning and contains an implementation of GCBS using `pl.LightningDataModule` and `pl.LightningModule`. It can be used for porting a PyTorch Lightning implementation to your training code.

## Profiling Script

We provide a profiling script for GCBS on random embeddings both for single gpu and multi gpu settings in `single_gpu_profiling_gcbs.py` and `multi_gpu_profiling_gcbs.py` respectively. These scripts contain a minimal implementation of just the algorithm and can be used for porting a PyTorch implementation to your training code.

## Minimal Implementation of GCBS in a PyTorch training loop 

We provide a minimal implementation the GCBS algorithm below along with the modifications to an existing PyTorch training loop at the beginning of an epoch. 

```python
import torch
import math
from scipy.sparse.csgraph import reverse_cuthill_mckee
from scipy.sparse import csr_matrix
import statistics
import torch.nn.functional.normalize as normalize

def compute_gcbs(z1, z2, quantile):
      # (1) Stack and normalize outputs
      src_train_full, tgt_train_full = normalize(z1).cuda(), normalize(z2).cuda()
      z1, z2 = [], []

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
      return list(reverse_cuthill_mckee(csr_matrix((data, (row, col)),
                                                    shape=(num_samples, num_samples))))

## Sample code within a Pytorch training loop (insert at the beginning of the epoch).
# model.eval()
# with torch.no_grad():
#     eval_z1, eval_z2 = [], []
#     for batch in train_dataloader:
#         #get paired embeddings for each batch
#         z1_inputs = batch[0].to(args.device), z2_inputs = batch[1].to(args.device)
#         z1_embedding, z2_embedding = model(z1_inputs), model(z2_inputs)

#         eval_z1.append(z1_embedding.cpu().float())
#         eval_z2.append(z2_embedding.cpu().float())

#     permutation = compute_gcbs(eval_z1, eval_z2, 0.999)
#     train_dataset = torch.utils.data.Subset(train_dataset, permutation)
#     train_sampler = SequentialSampler(train_dataset)
#     train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

# model.train()
# do standard training
```


## Citation
For citing this work please use the following Bibtex entry:

```
@InProceedings{pmlr-v202-sachidananda23a,
  title = 	 {Global Selection of Contrastive Batches via Optimization on Sample Permutations},
  author =       {Sachidananda, Vin and Yang, Ziyi and Zhu, Chenguang},
  booktitle = 	 {Proceedings of the 40th International Conference on Machine Learning},
  pages = 	 {29542--29562},
  year = 	 {2023},
  editor = 	 {Krause, Andreas and Brunskill, Emma and Cho, Kyunghyun and Engelhardt, Barbara and Sabato, Sivan and Scarlett, Jonathan},
  volume = 	 {202},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {23--29 Jul},
  publisher =    {PMLR},
  pdf = 	 {https://proceedings.mlr.press/v202/sachidananda23a/sachidananda23a.pdf},
  url = 	 {https://proceedings.mlr.press/v202/sachidananda23a.html},
  abstract = 	 {Contrastive Learning has recently achieved state-of-the-art performance in a wide range of unimodal and multimodal tasks. Many contrastive learning approaches use mined hard negatives to make batches more informative during training but these approaches are inefficient as they increase epoch length proportional to the number of mined negatives and require frequent updates of nearest neighbor indices or mining from recent batches. In this work, we provide an alternative to hard negative mining, Global Contrastive Batch Sampling (GCBS), an efficient approximation to the batch assignment problem that upper bounds the gap between the global and training losses, $\mathcal{L}^{Global} - \mathcal{L}^{Train}$, in contrastive learning settings. Through experimentation we find GCBS improves state-of-the-art performance in sentence embedding and code-search tasks. Additionally, GCBS is easy to implement as it requires only a few additional lines of code, does not maintain external data structures such as nearest neighbor indices, is more computationally efficient than the most minimal hard negative mining approaches, and makes no changes to the model being trained. Code is available at https://github.com/vinayak1/GCBS.}
}
```
