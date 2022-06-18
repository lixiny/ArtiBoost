from logging import log
from typing import Dict, List, Union, TypeVar

import networkx as nx
from anakin.utils.logger import logger
import torch.nn as nn

T = TypeVar("T", bound="Arch")


class Arch(nn.Module):
    def __init__(self: T, cfg: Dict, model_list: List[nn.Module]):
        super().__init__()
        self._model_list = nn.ModuleList(model_list)
        self._cfg = cfg
        self.parser()

    @property
    def model_list(self: T) -> nn.ModuleList:
        return self._model_list

    @property
    def models_params(self: T):
        # * should equal to model.parameters()
        # * see (https://pytorch.org/docs/stable/_modules/torch/nn/modules/module.html#Module) __setattr__
        return [{"params": filter(lambda p: p.requires_grad, m.parameters())} for m in self._model_list]

    def parser(self: T):
        items = self._cfg["ARCH"]
        self.models = {}
        if isinstance(items, Dict):
            items = [items]
        for i, item in enumerate(items):
            self.models[item["TYPE"]] = {"id": i, "previous": item["PREVIOUS"]}
        # use out-degree to find root
        outdegree = [0] * len(items)
        for _, v in self.models.items():
            for p in v["previous"]:
                outdegree[self.models[p]["id"]] += 1
        if outdegree.count(0) != 1:
            logger.error("Arch has multiple roots, a circle or other illegal input.!")
            raise Exception()
        self.root = items[outdegree.index(0)]["TYPE"]

    def to_graph(self: T):
        DG = nx.DiGraph()
        for k, v in self.models.items():
            DG.add_node(k)
            for p in v["previous"]:
                DG.add_edge(p, k)
        return DG

    # * uncomment this
    # def train(self: T, mode: bool = True) -> T:
    #     super(Arch, self).train(mode)

    def forward(self: T, input: Dict):
        self.outputs = {}
        self._forward(self.root, input)
        return self.outputs

    def _forward(self: T, mtype: str, input: Dict):
        # logger.debug(f"calculate {mtype}")
        inputs = dict(input)
        for p in self.models[mtype]["previous"]:
            if p not in self.outputs:
                self._forward(p, input)
            if len(inputs.keys() & self.outputs[p].keys()) > 0:
                logger.warning(f"key confilct! {inputs.keys() & self.outputs[p].keys()} will be rewrite!")
            inputs.update(self.outputs[p])
        self.outputs[mtype] = self._model_list[self.models[mtype]["id"]](inputs)
        # logger.debug(f"finish {mtype}")
