import logging
from anakin.utils import Registry, build_from_cfg, retrieve_from_cfg
from torch import nn

MODEL = Registry("model")
BACKBONE = Registry("backbone")
NECK = Registry("neck")
HEAD = Registry("head")
LOSS = Registry("loss")
DATASET = Registry("dataset")
METRIC = Registry("metric")

# ? builder function entrance
def build(cfg, registry, default_args=None):
    return build_from_cfg(cfg, registry, default_args)


def build_arch_model_list(cfg, preset_cfg, **kwargs):
    default_args = {
        "DATA_PRESET": preset_cfg,
    }
    for key, value in kwargs.items():
        default_args[key] = value

    if isinstance(cfg, list):
        models_list = [build_model(cfg_, default_args) for cfg_ in cfg]
        return models_list
    else:
        model = build_model(cfg, default_args)
        return [model]


def build_evaluator_metric_list(cfg, preset_cfg, **kwargs):
    default_args = {
        "DATA_PRESET": preset_cfg,
    }

    for key, value in kwargs.items():
        default_args[key] = value

    if isinstance(cfg, list):
        metric_list = [build_metric(cfg_, default_args) for cfg_ in cfg]
        return metric_list
    else:
        metric = build_metric(cfg, default_args)
        return [metric]


def build_criterion_loss_list(cfg, preset_cfg, **kwargs):
    # repacking preset_cfg & kwargs
    default_args = {
        "DATA_PRESET": preset_cfg,
    }
    for key, value in kwargs.items():
        default_args[key] = value

    if isinstance(cfg, list):
        loss_list = [build_loss(cfg_, default_args) for cfg_ in cfg]
        return loss_list
    else:
        loss = build_loss(cfg, default_args)
        return [loss]


def build_dataset(cfg, preset_cfg, **kwargs):
    exec(f"from ..datasets import {cfg['TYPE']}")
    default_args = {
        "DATA_PRESET": preset_cfg,
    }

    for key, value in kwargs.items():
        default_args[key] = value
    return build(cfg, DATASET, default_args=default_args)


def build_metric(cfg, default_args=None):
    metric = build(cfg, METRIC, default_args=default_args)
    return metric


def build_model(cfg, default_args=None):
    exec(f"from ..models import {cfg['TYPE']}")
    model = build(cfg, MODEL, default_args=default_args)
    return model


def build_head(cfg, default_args=None):
    return build(cfg, HEAD, default_args=default_args)


def build_neck(cfg, default_args=None):
    return build(cfg, NECK, default_args=default_args)


def build_backbone(cfg, default_args=None):
    return build(cfg, BACKBONE, default_args=default_args)


def build_loss(cfg, default_args=None):
    return build(cfg, LOSS, default_args=default_args)
