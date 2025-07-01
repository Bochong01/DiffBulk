# DiffBulk: Enhancing Spatial Transcriptomic Predication with Diffusion-Based Training
DiffBulk introduces a two-stage framework for learning gene-aware image representations via conditional diffusion modeling.

## Quick Start
1. Clone the repository
```bash
git clone https://github.com/Bochong01/DiffBulk.git
cd DiffBulk
```

2. Diffusion Pretraining
```bash
cd Pretrain
bash train.sh
```

3. Post EMA
```bash
bash ema.sh
```

4. Downstream Gene Expression Training
Before starting downstream gene expression prediction training, ensure you have set hyperparameters in `config.yaml`:
- <GENE_DIM>: The number of genes to be detected.
```bash
cd DiffBulk/Downstream
bash train.sh
```
