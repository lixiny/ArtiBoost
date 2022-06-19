# design
# first each (compatible) dataset has its own epoch pass
# the epoch pass has default params
# the epoch pass have 3 flags
# -
from __future__ import annotations
from abc import ABCMeta, abstractmethod
from typing import Optional, Union, Mapping, MutableMapping, Type, Any, Callable, Tuple, Sequence
import torch
import numpy as np

from anakin.criterions.criterion import Criterion
from anakin.metrics.evaluator import Evaluator
from anakin.models.arch import Arch
from anakin.postprocess.iknet.fittingunit import FittingUnit
from anakin.viztools.opendr_renderer import OpenDRRenderer

from anakin.opt import arg


class SubmitEpochPass(metaclass=ABCMeta):
    """"""

    build_reg: MutableMapping[str, Type[SubmitEpochPass]] = {}

    @staticmethod
    def build(key: str, **kwargs) -> SubmitEpochPass:
        return SubmitEpochPass.build_reg[key](**kwargs)

    @staticmethod
    def reg(key: str) -> Callable[[Type[SubmitEpochPass]], Type[SubmitEpochPass]]:
        def fn(cls: Type[SubmitEpochPass]) -> Type[SubmitEpochPass]:
            SubmitEpochPass.build_reg[key] = cls
            return cls

        return fn

    def __init__(self, cfg, ):  # TODO
        # self.true_root: bool = cfg["TRUE_ROOT"]  # TODO move to freihand
        self.dump: bool = arg.submit_dump
        self.fit_mesh: bool = arg.postprocess_fit_mesh
        self.fit_mesh_ik: str = arg.postprocess_fit_mesh_ik
        self.fit_mesh_use_fitted_joints = arg.postprocess_fit_mesh_use_fitted_joints
        self.postprocess_draw: bool = arg.postprocess_draw
        # self.postprocess_draw_path: Optional[str] = arg.postprocess_draw_path

        if self.fit_mesh:
            self.fitting_unit = FittingUnit()
        else:
            self.fitting_unit = None

        if self.postprocess_draw:
            self.renderer = OpenDRRenderer()
        else:
            self.renderer = None

        self.sample_counter = 0

    def reset_sample_counter(self):
        self.sample_counter = 0

    # TODO: Currently, there just has no meaning to override this method
    # TODO: just use inheritence for consistency of code style
    def mesh_fit(
        self, inp: Mapping[str, Any], pred_joints: torch.Tensor
    ) -> Tuple[Sequence[np.ndarray], Sequence[np.ndarray]]:
        return self.fitting_unit(inp, pred_joints)

    @abstractmethod
    def draw_batch(self, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def dump_json(self, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def __call__(
        self,
        epoch_idx: int,
        data_loader: Optional[torch.utils.data.DataLoader],
        arch_model: Union[Arch, torch.nn.parallel.DistributedDataParallel, torch.nn.DataParallel],
        criterion: Optional[Criterion],
        evaluator: Optional[Evaluator],
        rank: int,
        dump_path: str,
    ):
        raise NotImplementedError()
