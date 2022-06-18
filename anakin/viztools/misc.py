from anakin.utils.misc import ImmutableClass


class CONSTANTS(metaclass=ImmutableClass):
    colors = {
        "colors": [228 / 255, 178 / 255, 148 / 255],
        "light_pink": [0.9, 0.7, 0.7],  # This is used to do no-3d
        "light_blue": [102 / 255, 209 / 255, 243 / 255],
    }

    color_hand_joints = [
        [1.0, 0.0, 0.0],
        [0.0, 0.4, 0.0],
        [0.0, 0.6, 0.0],
        [0.0, 0.8, 0.0],
        [0.0, 1.0, 0.0],  # thumb
        [0.0, 0.0, 0.6],
        [0.0, 0.0, 1.0],
        [0.2, 0.2, 1.0],
        [0.4, 0.4, 1.0],  # index
        [0.0, 0.4, 0.4],
        [0.0, 0.6, 0.6],
        [0.0, 0.8, 0.8],
        [0.0, 1.0, 1.0],  # middle
        [0.4, 0.4, 0.0],
        [0.6, 0.6, 0.0],
        [0.8, 0.8, 0.0],
        [1.0, 1.0, 0.0],  # ring
        [0.4, 0.0, 0.4],
        [0.6, 0.0, 0.6],
        [0.8, 0.0, 0.8],
        [1.0, 0.0, 1.0],
    ]  # little

    mayavi_cache_path = "/tmp/anakin/mayavi"