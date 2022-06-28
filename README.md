<br />
<p align="center">
    <img src="docs/artiboost_logo_512ppi.png" alt="icon" width="40%">
    <h1 align="center">
        Boosting Articulated 3D Hand-Object Pose Estimation via Online Exploration and Synthesis 
    </h1>

  <p align="center">
    <img src="docs/capture.png"" alt="capture" width="75%">
  </p>
  <p align="center">
    <strong>CVPR, 2022</strong>
    <br />
    <a href="https://lixiny.github.io"><strong>Lixin Yang * </strong></a>
    .
    <a href="https://kailinli.top"><strong>Kailin Li *</strong></a>
    ·
    <a href=""><strong>Xinyu Zhan</strong></a>
    ·
    <a href="https://lyuj1998.github.io"><strong>Jun Lv</strong></a>
    ·
    <a href=""><strong>Wenqiang Xu</strong></a>
    ·
    <a href="https://jeffli.site"><strong>Jiefeng Li</strong></a>
    ·
    <a href="https://mvig.sjtu.edu.cn"><strong>Cewu Lu</strong></a>
    <br />
    \star = equal contribution
  </p>
  
  <p align="center">
    <a href='https://openaccess.thecvf.com/content/CVPR2022/html/Yang_ArtiBoost_Boosting_Articulated_3D_Hand-Object_Pose_Estimation_via_Online_Exploration_CVPR_2022_paper.html'>
      <img src='https://img.shields.io/badge/Paper-PDF-green?style=flat&logo=Googlescholar&logoColor=blue' alt='Paper PDF'>
    </a>
    <a href='https://arxiv.org/abs/2109.05488' style='padding-left: 0.5rem;'>
      <img src='https://img.shields.io/badge/ArXiv-PDF-green?style=flat&logo=arXiv&logoColor=green' alt='ArXiv PDF'>
    </a>
    <a href='#' style='padding-left: 0.5rem;'>
      <img src='https://img.shields.io/badge/Project-Page-blue?style=flat&logo=Google%20chrome&logoColor=blue' alt='Project Page'>
    <a href='https://www.youtube.com/watch?v=QbPsjWRyloY' style='padding-left: 0.5rem;'>
      <img src='https://img.shields.io/badge/Youtube-Video-red?style=flat&logo=youtube&logoColor=red' alt='Youtube Video'>
    </a>
  </p>
</p>

<br />

This repo contains models, train, and test code.

## TODO

- [ ] installation guideline
- [x] testing code and pretrained models (uploading)
- [ ] generating CCV-space
- [ ] training pipeline

## Installation

<a href="https://pytorch.org/get-started/locally/">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-1.8.1-ee4c2c?logo=pytorch&logoColor=red">
</a>
<a href="https://developer.nvidia.com/cuda-11.1.0-download-archive" style='padding-left: 0.1rem;'>
  <img alt="PyTorch" src="https://img.shields.io/badge/CUDA-11.1-yellow?logo=nvidia&logoColor=yellow">
</a>
<a href="https://releases.ubuntu.com/20.04/" style='padding-left: 0.1rem;'>
  <img alt="Ubuntu" src="https://img.shields.io/badge/Ubuntu-20.04-green?logo=ubuntu&logoColor=yelgreenlow">
</a>

Following the [Installation Instruction](docs/Installation.md) to setup environment, assets, datasets and models.

## Evaluation

### Heatmap-based model, HO3Dv2

```shell
$ python train/submit_reload.py --cfg config_eval/eval_ho3dv2_clasbased_artiboost.yaml --gpu_id 0 --submit_dump
```

This script yield the [Ours *Clas* + **Arti**] results in main paper Table 2. HO3Dv2 codalab submission file will be dumped at `common/eval_ho3dv2_clasbased_artiboost_SUBMIT.json`

### Regression-based model, HO3Dv2

```shell
$ python train/submit_reload.py --cfg config_eval/eval_ho3dv2_regbased_artiboost.yaml --gpu_id 0 --submit_dump
```

This script yield the [Ours *Reg* + **Arti**] results in main paper Table 2.

### Heatmap-based model, HO3Dv3

```shell
$ python train/submit_reload.py --cfg config_eval/eval_ho3dv3_clasbased_artiboost.yaml --gpu_id 0 --submit_dump
```

This script yield the [Ours *Clas* + **Arti**] results in main paper Table 5.

### Heatmap-based model, HO3Dv3, Symmetry model

```shell
$ python train/submit_reload.py --cfg config_eval/eval_ho3dv3_clasbased_sym_artiboost.yaml --gpu_id 0 --submit_dump
```

This script yield the [Ours *Clas* sym + **Arti**] results in main paper Table 5.

### DexYCB ...

## Generate CCV

## Training Pipeline

## MANO Driver

## Acknowledge & Citation

```
@inproceedings{li2021artiboost,
    title={{ArtiBoost}: Boosting Articulated 3D Hand-Object Pose Estimation via Online Exploration and Synthesis},
    author={Li, Kailin and Yang, Lixin and Zhan, Xinyu and Lv, Jun and Xu, Wenqiang and Li, Jiefeng and Lu, Cewu},
    booktitle={arXiv preprint arXiv:2109.05488},
    year={2021}
}
```
