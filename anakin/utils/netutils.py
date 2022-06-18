from typing import Iterable

import torch
import transformers
from torch.optim import Optimizer


def freeze_batchnorm_stats(model):
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.momentum = 0
    for name, child in model.named_children():
        freeze_batchnorm_stats(child)


def recurse_freeze(model):
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.momentum = 0
    for name, child in model.named_children():
        for param in child.parameters():
            param.requires_grad = False
        recurse_freeze(child)


def build_optimizer(params: Iterable, **cfg):
    if cfg["OPTIMIZER"] == "Adam" or cfg["OPTIMIZER"] == "adam":
        return torch.optim.Adam(
            params,
            lr=cfg["LR"],
            weight_decay=float(cfg.get("WEIGHT_DECAY", 0.0)),
        )
    elif cfg["OPTIMIZER"] == "SGD" or cfg["OPTIMIZER"] == "sgd":
        return torch.optim.SGD(
            params,
            lr=cfg["LR"],
            momentum=float(cfg.get("MOMENTUM", 0.0)),
            weight_decay=float(cfg.get("WEIGHT_DECAY", 0.0)),
        )
    else:
        raise NotImplementedError(f"{cfg['OPTIMIZER']} not yet be implemented")


def build_scheduler(optimizer: Optimizer, **cfg):
    scheduler = cfg.get("SCHEDULER", "StepLR")
    if scheduler == "StepLR":
        return torch.optim.lr_scheduler.StepLR(optimizer, cfg["LR_DECAY_STEP"], gamma=cfg["LR_DECAY_GAMMA"])
    elif scheduler == "constant_warmup":
        return transformers.get_constant_schedule_with_warmup(optimizer, num_warmup_steps=cfg["NUM_WARMUP_STEPS"])
    elif scheduler == "cosine_warmup":
        return transformers.get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=cfg["NUM_WARMUP_STEPS"],
            num_training_steps=cfg["NUM_TRAINING_STEPS"],
        )
    elif scheduler == "linear_warmup":
        return transformers.get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=cfg["NUM_WARMUP_STEPS"],
            num_training_steps=cfg["NUM_TRAINING_STEPS"],
        )
    else:
        raise NotImplementedError(f"{scheduler} not yet be implemented")
