# Installation

## Environment

### Get code

```shell
$ cd ${HOME}
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

finally, at the directory: `${HOME}/ArtiBoost/thirdparty/`, use pip install

```shell
$ pip install ./dex-ycb-toolkit
```

verify:

```shell
$ python -c "from dex_ycb_toolkit.dex_ycb import DexYCBDataset, _YCB_CLASSES"
```

## Datasets

## Data assets

## Pretrained models
