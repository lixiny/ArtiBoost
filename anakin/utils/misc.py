import functools
import math
import yaml
import re
import numpy as np
from enum import Enum
from collections import namedtuple

from termcolor import colored

RandomState = namedtuple(
    "RandomState",
    [
        "torch_rng_state",
        "torch_cuda_rng_state",
        "torch_cuda_rng_state_all",
        "numpy_rng_state",
        "random_rng_state",
    ],
)
RandomState.__new__.__default__ = (None,) * len(RandomState._fields)


class TrainMode(Enum):
    TRAIN = 0
    VAL = 1
    TEST = 2


def enable_lower_param(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        kw_uppers = {}
        for k, v in kwargs.items():
            kw_uppers[k.upper()] = v
        return func(*args, **kw_uppers)

    return wrapper


def singleton(cls):
    _instance = {}

    @functools.wraps(cls)
    def inner(*args, **kwargs):
        if cls not in _instance:
            _instance[cls] = cls(*args, **kwargs)
        return _instance[cls]

    return inner


class ImmutableClass(type):
    def __call__(cls, *args, **kwargs):
        raise AttributeError("Cannot instantiate this class")

    def __setattr__(cls, name, value):
        raise AttributeError("Cannot modify immutable class")

    def __delattr__(cls, name):
        raise AttributeError("Cannot delete immutable class")


class CONST(metaclass=ImmutableClass):
    PI = math.pi
    INT_MAX = 2 ** 32 - 1
    NUM_JOINTS = 21
    NUM_CORNERS = 8
    SIDE = "right"
    DUMMY = "dummy"
    JOINTS_IDX_PARENTS = [0, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19]
    CORNERCUBE_IDX_ORDER = [
        (0, 1),
        (0, 2),
        (1, 3),
        (2, 3),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
        (4, 5),
        (4, 6),
        (5, 7),
        (6, 7),
    ]
    REF_BONE_LEN = 0.09473151311686484  # in meter
    PYRENDER_EXTRINSIC = np.array(
        [
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1],
        ],
        dtype=np.float32,
    )

    YCB_IDX2CLASSES = {
        1: "002_master_chef_can",
        2: "003_cracker_box",
        3: "004_sugar_box",
        4: "005_tomato_soup_can",
        5: "006_mustard_bottle",
        6: "007_tuna_fish_can",
        7: "008_pudding_box",
        8: "009_gelatin_box",
        9: "010_potted_meat_can",
        10: "011_banana",
        11: "019_pitcher_base",
        12: "021_bleach_cleanser",
        13: "024_bowl",
        14: "025_mug",
        15: "035_power_drill",
        16: "036_wood_block",
        17: "037_scissors",
        18: "040_large_marker",
        19: "051_large_clamp",
        20: "052_extra_large_clamp",
        21: "061_foam_brick",
    }


def update_config(config_file):
    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        return config

def format_args(args, cfg={}):
    args_list = [f" - {colored(name, 'green')}: {getattr(args, name)}" for name in vars(args)]
    cfg_list = [f" - {colored(k, 'magenta')}: {v}" for k, v in cfg.items()]
    return "\n".join(args_list+cfg_list)


def param_count(net):
    return sum(p.numel() for p in net.parameters()) / 1e6


def param_size(net):
    # ! treat all parameters to be float
    return sum(p.numel() for p in net.parameters()) * 4 / (1024 * 1024)


camel_to_snake_pattern = re.compile(r"(?<!^)(?=[A-Z])")


def camel_to_snake(name: str) -> str:
    return camel_to_snake_pattern.sub("_", name).lower()
