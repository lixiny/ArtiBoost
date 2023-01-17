# Installation

## Environment

### 1. Get code

```shell
$ git clone https://github.com/lixiny/ArtiBoost.git
$ cd ArtiBoost
```

### 2. Set up new environment:

```sh
$ conda env create -f environment.yml
$ conda activate artiboost
```

### 3. Install dependencies

```sh
# inside your artiboost env
$ pip install -r requirements.txt
```

### 4. Install thirdparty

- **dex-ycb-toolkit**

  ```shell
  $ cd thirdparty
  $ git clone --recursive https://github.com/NVlabs/dex-ycb-toolkit.git
  ```

  We need install **dex-ycb-toolkit** as a python package. Following the steps:

  1. you need to install:

     ```shell
     $ sudo apt-get install liboctomap-dev
     $ sudo apt-get install libfcl-dev

     # or delete the `python-fcl` in dex-ycb-toolkit/setup.py
     ```

  2. create a `__init__.py` in _dex_ycb_toolkit_

     ```shell
     $ cd thirdparty/dex-ycb-toolkit/dex_ycb_toolkit/
     $ touch __init__.py
     ```

  3. change a line in `dex-ycb-toolkit/setup.py`:
     ```
     line #16:  opencv-python ==> opencv-python-headless
     ```

  finally, at the directory: `./thirdparty`, use pip install

  ```sh
  # inside your artiboost env
  $ pip install ./dex-ycb-toolkit
  ```

  to verify:

  ```shell
  $ python -c "from dex_ycb_toolkit.dex_ycb import DexYCBDataset, _YCB_CLASSES"
  ```

## Datasets

### HO3D

Download HO3D [**v2**](https://arxiv.org/abs/1907.01481.pdf) and [**v3**](https://arxiv.org/abs/2107.00887) from the [official site](https://www.tugraz.at/index.php?id=40231). Then unzip and link the datasets in `./data`.  
Now your `./data` folder should have structure like:

```
    ├── HO3D
    │   ├── evaluation
    │   ├── evaluation.txt
    │   ├── train
    │   └── train.txt
    ├── HO3D_v3
    │   ├── calibration
    │   ├── evaluation
    │   ├── evaluation.txt
    │   ├── manual_annotations
    │   ├── train
    │   └── train.txt
```

### DexYCB

Download [DexYCB](https://arxiv.org/abs/2104.04631) dataset from the [official site](https://dex-ycb.github.io). Then unzip and link the dataset in `./data`.  
Your `./data` folder should have structure like:

```
    ...
    ├── DexYCB
    │   ├── 20200709-subject-01
    │   ├── 20200813-subject-02
    │   ├── 20200820-subject-03
    │   ├── 20200903-subject-04
    │   ├── 20200908-subject-05
    │   ├── 20200918-subject-06
    │   ├── 20200928-subject-07
    │   ├── 20201002-subject-08
    │   ├── 20201015-subject-09
    │   ├── 20201022-subject-10
    │   ├── bop
    │   ├── calibration
    │   └── models
```

### YCB Object Models

Download our pre-processed YCB objects from:

- [YCB_models_supp](https://drive.google.com/file/d/1v36yY5AOSRO1nN42e8y0bwRAm2rs2QbW/view?usp=share_link)
- [YCB_models_process](https://drive.google.com/file/d/1lg4_GK3Ztmk6fd0q3cdiqQX-iAqMwFVm/view?usp=share_link)

then unzip and copy them to your `./data`.

### HTML Hand Texture Model

Download our pre-process hand .obj with textures from:

- [HTML_supp](https://drive.google.com/file/d/1GnQbJJa1OZlnBKUmv1pK7wvUjdr1bzhT/view?usp=share_link)

(optional) Download HTML hand texture model from the [official site](https://handtracker.mpi-inf.mpg.de/projects/HandTextureModel/).  
then unzip and copy them into `./data`.

---

Finally, you will have `./data` with structure like:

```
    ├── DexYCB
    ├── HO3D
    ├── HO3D_v3
    ├── HTML_release
    │   ├── HTML__hello_world.py
    │   └── ...
    ├── HTML_supp
    │   ├── html_001
    │   ├── ...
    │   ├── html.obj
    │   └── html.obj.mtl
    ├── YCB_models_process
    │   ├── 002_master_chef_can
    │   └── ...
    └── YCB_models_supp
        ├── 002_master_chef_can
        └── ...
```

## Data Assets

Data assets are essential for ArtiBoost training and evaluation.  
Download the **assets** folder at [here](https://drive.google.com/drive/folders/189JIJ7NDUNI1jXu1fVHgbjhYbce-xjB5?usp=share_link) and place it as `./assets`.

The `./assets` folder should contains:

- `GrabNet/`: GrabNet model's weights.  
   It is a copy of **_GrabNet model files/weights_** from [GRAB](https://grab.is.tue.mpg.de/index.html) [_Taheri etal ECCV2020_]
- `hasson20_assets/`:  
   This folder contains essentials to run our [honetMANO](../anakin/models/honetMANO.py) on FPHAB dataset.  
   It is a copy of _**assets**_ folder in [handobjectconsist](https://github.com/hassony2/handobjectconsist) [_Hasson etal CVPR2020_].

- `postprocess/`:  
   [IKNet](../anakin/postprocess/iknet/model.py) model's weights. Convert hand joints position to MANO rotations.  
   This checkpoints is trained in the original [HandTailor](https://github.com/LyuJ1998/HandTailor) [_Lv etal BMVC2021_]

- `mano_v1_2/`: MANO hand model.  
   Download **_Models & Code_** at [MANO website](https://mano.is.tue.mpg.de/download.php). Then unzip the downloaded file: _mano_v1_2.zip_.

- `ho3d_corners.pkl`: HO3D object corner's annotation.
- `extend_models_info.json`: YCB objects' principal axis of inertia.  
   For evaluating maximum symmetry-aware surface distance (MSSD).
