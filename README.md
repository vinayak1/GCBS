# Global Contrastive Batch Sampling
Repository for the ICML 2023 paper "Global Selection of Contrastive Batches via Optimization on Sample Permutations" [https://proceedings.mlr.press/v202/sachidananda23a.html]

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
