# Installation

## Environment

### Get code

```shell
$ git clone https://github.com/lixiny/ArtiBoost.git
$ cd ArtiBoost
```

### Set up new environment:

```shell script
$ conda env create -f environment_release.yml
$ conda activate artiboost
```

### Install dependencies

```shell script
$ pip install -r requirements.txt
```

### Install thirdparty

- **dex-ycb-toolkit**

```shell
$ cd thirdparty
$ git clone --recursive https://github.com/NVlabs/dex-ycb-toolkit.git
```

We need install **dex-ycb-toolkit** as a python package.  
first, you need to install:

```shell
$ sudo apt-get install liboctomap-dev
$ sudo apt-get install libfcl-dev

# or delete the `python-fcl` in dex-ycb-toolkit/setup.py
```

second, create a `__init__.py` in `dex_ycb_toolkit`

```shell
$ cd thirdparty/dex-ycb-toolkit/dex_ycb_toolkit/
$ touch __init__.py
```

finally, at the directory: `./thirdparty`, use pip install

```shell
$ pip install ./dex-ycb-toolkit
```

verify:

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
    ├── HO3D
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

- [YCB_models_supp](https://www.dropbox.com/s/psp18fxlcx92k4d/YCB_models_supp.zip?dl=0)
- [YCB_models_process](https://www.dropbox.com/s/vukf9hr8zibcs6n/YCB_models_process.zip?dl=0)

then unzip and copy them to your `./data`.

### HTML Hand Texture Model

Download our pre-process hand .obj with textures from:

- [HTML_supp](https://www.dropbox.com/s/8k4c0qq0b3rjpsc/HTML_supp.zip?dl=0)

(optional) Download HTML hand texture model from the [official site](https://handtracker.mpi-inf.mpg.de/projects/HandTextureModel/).  
then unzip and copy them into `./data`.

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

## Pretrained models
