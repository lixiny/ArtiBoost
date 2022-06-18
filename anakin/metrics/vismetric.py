import io
from abc import ABC, abstractmethod
from typing import Dict, List

import cv2
import numpy as np
import torch
from anakin.datasets.hoquery import Queries
from anakin.metrics.metric import AverageMeter, Metric
from anakin.utils.builder import METRIC
from anakin.utils.logger import logger
from anakin.utils.misc import CONST
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image


class VisMetric(Metric, ABC):
    def __init__(self, **cfg) -> None:
        super().__init__()
        self._image = None

    def reset(self):
        self.count = 0

    @property
    def image(self):
        return self._image

    @image.setter
    def image(self, new_image):
        self._image = new_image

    @abstractmethod
    def feed(self, preds: Dict, targs: Dict, **kwargs):
        pass

    def get_measures(self, **kwargs) -> Dict:
        raise NotImplementedError()

    @staticmethod
    def _squashfig(fig=None):
        # TomNorway - https://stackoverflow.com/a/53516034
        if not fig:
            fig = plt.gcf()

        fig.subplots_adjust(0, 0, 1, 1, 0, 0)
        for ax in fig.axes:
            if isinstance(ax, Axes3D):
                ax.margins(0, 0, 0)
                ax.xaxis.set_major_locator(plt.NullLocator())
                ax.yaxis.set_major_locator(plt.NullLocator())
                ax.zaxis.set_major_locator(plt.NullLocator())
            else:
                ax.axis("off")
                ax.margins(0, 0)
                ax.xaxis.set_major_locator(plt.NullLocator())
                ax.yaxis.set_major_locator(plt.NullLocator())

    @staticmethod
    def fig2image(fig):
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        im = np.array(Image.open(buf).convert("RGB"))[:, :, (2, 1, 0)]
        buf.close()
        return im


@METRIC.register_module
class Vis2DMetric(VisMetric):
    def __init__(self, **cfg) -> None:
        super().__init__(**cfg)
        plt.switch_backend("agg")
        self.fig = plt.figure(figsize=(10, 10))
        self.inp_res = cfg["DATA_PRESET"]["IMAGE_SIZE"]
        self.ncol = cfg.get("NCOL", 3)
        self.nrow = cfg.get("NROW", 3)
        self.corner_link_order = np.array(cfg.get("CORNER_LINK_ORDER", list(range(8))))

    def feed(self, preds: Dict, targs: Dict, **kwargs):
        batch_size = targs[Queries.IMAGE].shape[0]
        if self.count > 0:
            self.count += batch_size
            return

        # ==== prepare data >>>>
        image = targs[Queries.IMAGE].permute(0, 2, 3, 1).cpu() + 0.5
        if "2d_uvd" in preds:
            pred_uvd_joints = preds["2d_uvd"][:, : CONST.NUM_JOINTS, :].detach().cpu()
            pred_uvd_corners = preds["2d_uvd"][:, CONST.NUM_JOINTS :, :].detach().cpu()
            inp_res = torch.Tensor(self.inp_res).cpu()
            pred_joints_2d = torch.einsum("bij,j->bij", pred_uvd_joints[:, :, :2], inp_res)
            pred_corners_2d = torch.einsum("bij,j->bij", pred_uvd_corners[:, :, :2], inp_res)
        else:
            pred_joints_2d = preds["joints_2d"].detach().cpu()
            pred_corners_2d = preds["corners_2d"].detach().cpu()

        # pred_joints_2d = torch.bmm(targs["cam_intr"], preds["joints_3d_abs"].detach().cpu().transpose(1, 2)).transpose(
        #     1, 2
        # )
        # pred_joints_2d = pred_joints_2d[:, :, :2] / pred_joints_2d[:, :, [2]]
        # pred_corners_2d = torch.bmm(targs["cam_intr"], preds["corners_3d_abs"].detach().cpu().transpose(1, 2)).transpose(
        #     1, 2
        # )
        # pred_corners_2d = pred_corners_2d[:, :, :2] / pred_corners_2d[:, :, [2]]

        # assert torch.allclose(pred_joints_2d_uvd, pred_joints_2d)
        # assert torch.allclose(pred_corners_2d_uvd[:, :8, :], pred_corners_2d)

        joints_2d = targs[Queries.JOINTS_2D][:, : CONST.NUM_JOINTS, :].cpu()
        corners_2d = targs[Queries.CORNERS_2D].cpu()

        joints_vis = targs[Queries.JOINTS_VIS].cpu()
        corners_vis = targs[Queries.CORNERS_VIS].cpu()
        gt_root = joints_2d[:, 0, :]

        # ==== draw pred and gt >>>>
        pred_image = self.draw_batch_joints_image(
            batch_images=image,
            batch_joints2d=pred_joints_2d,
            batch_corners2d=pred_corners_2d,
            batch_gt_root2d=gt_root,
            batch_joints_vis=joints_vis,
            batch_corners_vis=corners_vis,
        )
        ref_image = self.draw_batch_joints_image(
            batch_images=image,
            batch_joints2d=joints_2d,
            batch_corners2d=corners_2d,
            batch_gt_root2d=gt_root,
            batch_joints_vis=joints_vis,
            batch_corners_vis=corners_vis,
        )
        self.image = np.concatenate([pred_image, ref_image], axis=1)
        self.count += batch_size

    def draw_batch_joints_image(
        self,
        batch_images,
        batch_joints2d,
        batch_corners2d=None,
        batch_gt_root2d=None,
        batch_joints_vis=None,
        batch_corners_vis=None,
    ):
        batch_size = batch_images.shape[0]
        self.fig.clf()
        axes = self.fig.subplots(self.nrow, self.ncol)
        for i in range(self.ncol * self.nrow):
            if i >= batch_size:
                continue
            row = i // self.ncol
            col = i % self.ncol
            # >>>> print image >>>>
            axes[row, col].imshow(batch_images[i])
            axes[row, col].axis("off")
            # >>>> print joints >>>>
            j = batch_joints2d[i]
            # self.visualize_joints_2d(axes[row, col], j, joint_idxs=False)
            self.plot_hand(axes[row, col], j)
            # >>>> print corners >>>>
            try:
                c = batch_corners2d[i]
                self.visualize_joints_2d(
                    axes[row, col],
                    c,
                    alpha=1,
                    joint_idxs=False,
                    point_color="turquoise",
                    links=[
                        [],
                        [],
                        [],
                        [],
                        [],
                        self.corner_link_order[[0, 1, 3, 2, 0]],
                        self.corner_link_order[[4, 5, 7, 6, 4]],
                        self.corner_link_order[[1, 5]],
                        self.corner_link_order[[2, 6]],
                        self.corner_link_order[[3, 7]],
                        self.corner_link_order[[0, 4]],
                    ],
                )
            except:
                pass
            # >>>> print GT root >>>>
            if batch_gt_root2d is not None:
                gt_root = batch_gt_root2d[i]
                axes[row, col].scatter(gt_root[0], gt_root[1], 150, "blueviolet", "*")

            try:
                axes[row, col].scatter(c[0][0], c[0][1], 150, "red", "^")
                axes[row, col].scatter(c[7][0], c[7][1], 150, "yellow", "^")
            except:
                pass

            try:
                j_vis = batch_joints_vis[i]
                for j_idx, j_v in enumerate(j_vis):
                    if not bool(j_v):
                        axes[row, col].scatter(j[j_idx][0], j[j_idx][1], 125, "gray")
                c_vis = batch_corners_vis[i]
                for c_idx, c_v in enumerate(c_vis):
                    if not bool(c_v):
                        axes[row, col].scatter(c[c_idx][0], c[c_idx][1], 125, "gray")
            except:
                pass

        self._squashfig(self.fig)
        image = self.fig2image(self.fig)
        return image

    @staticmethod
    def visualize_joints_2d(
        ax,
        joints,
        joint_idxs=True,
        links=None,
        alpha=1,
        scatter=True,
        linewidth=2,
        point_color=None,
        color=None,
        axis_equal=True,
    ):
        if links is None:
            links = [
                (0, 1, 2, 3, 4),
                (0, 5, 6, 7, 8),
                (0, 9, 10, 11, 12),
                (0, 13, 14, 15, 16),
                (0, 17, 18, 19, 20),
            ]
        x = joints[:, 0]
        y = joints[:, 1]
        if scatter:
            ax.scatter(x, y, 30, "r" if not point_color else point_color)

        for row_idx, row in enumerate(joints):
            if joint_idxs:
                plt.annotate(str(row_idx), (row[0], row[1]))
        Vis2DMetric._draw2djoints(ax, joints, links, alpha=alpha, linewidth=linewidth, color=color)
        if axis_equal:
            ax.axis("equal")

    @staticmethod
    def _draw2djoints(ax, annots, links, alpha=1, linewidth=1, color=None):
        colors = [
            "firebrick",
            "orangered",
            "orange",
            "gold",
            "violet",
            "seagreen",
            "steelblue",
            "forestgreen",
            "dodgerblue",
            "deepskyblue",
            "royalblue",
        ]

        for finger_idx, finger_links in enumerate(links):
            for idx in range(len(finger_links) - 1):
                if color is not None:
                    link_color = color[finger_idx]
                else:
                    link_color = colors[finger_idx]
                Vis2DMetric._draw2dseg(
                    ax,
                    annots,
                    finger_links[idx],
                    finger_links[idx + 1],
                    c=link_color,
                    alpha=alpha,
                    linewidth=linewidth,
                )

    @staticmethod
    def _draw2dseg(ax, annot, idx1, idx2, c="r", alpha=1, linewidth=1):
        ax.plot(
            [annot[idx1, 0], annot[idx2, 0]],
            [annot[idx1, 1], annot[idx2, 1]],
            c=c,
            alpha=alpha,
            linewidth=linewidth,
        )

    @staticmethod
    def plot_hand(axis, coords_hw, linewidth="3"):
        coords_hw = coords_hw.numpy()
        colors = np.array(
            [
                [1.0, 0.0, 0.0],
                #
                [0.0, 0.4, 0.2],
                [0.0, 0.6, 0.3],
                [0.0, 0.8, 0.4],
                [0.0, 1.0, 0.5],
                #
                [0.0, 0.0, 0.4],
                [0.0, 0.0, 0.6],
                [0.0, 0.0, 0.8],
                [0.0, 0.0, 1.0],
                #
                [0.0, 0.4, 0.4],
                [0.0, 0.6, 0.6],
                [0.0, 0.8, 0.8],
                [0.0, 1.0, 1.0],
                #
                [0.4, 0.4, 0.0],
                [0.6, 0.6, 0.0],
                [0.8, 0.8, 0.0],
                [1.0, 1.0, 0.0],
                #
                [0.4, 0.0, 0.4],
                [0.6, 0.0, 0.6],
                [0.7, 0.0, 0.8],
                [1.0, 0.0, 1.0],
            ]
        )

        colors = colors[:, ::-1]

        # define connections and colors of the bones
        bones = [
            ((0, 1), colors[1, :]),
            ((1, 2), colors[2, :]),
            ((2, 3), colors[3, :]),
            ((3, 4), colors[4, :]),
            ((0, 5), colors[5, :]),
            ((5, 6), colors[6, :]),
            ((6, 7), colors[7, :]),
            ((7, 8), colors[8, :]),
            ((0, 9), colors[9, :]),
            ((9, 10), colors[10, :]),
            ((10, 11), colors[11, :]),
            ((11, 12), colors[12, :]),
            ((0, 13), colors[13, :]),
            ((13, 14), colors[14, :]),
            ((14, 15), colors[15, :]),
            ((15, 16), colors[16, :]),
            ((0, 17), colors[17, :]),
            ((17, 18), colors[18, :]),
            ((18, 19), colors[19, :]),
            ((19, 20), colors[20, :]),
        ]

        for connection, color in bones:

            coord1 = coords_hw[connection[0], ::-1]
            coord2 = coords_hw[connection[1], ::-1]
            coords = np.stack([coord1, coord2])

            axis.plot(coords[:, 1], coords[:, 0], color=color, linewidth=linewidth, markersize=10)
        for i in range(coords_hw.shape[0]):
            axis.plot(coords_hw[i, 0], coords_hw[i, 1], "o", color=colors[i, :], markersize=6)


@METRIC.register_module
class VisHand2DMetric(Vis2DMetric):
    def feed(self, preds: Dict, targs: Dict, **kwargs):
        batch_size = targs[Queries.IMAGE].shape[0]
        if self.count > 0:
            self.count += batch_size
            return

        # ==== prepare data >>>>
        image = targs[Queries.IMAGE].permute(0, 2, 3, 1).cpu() + 0.5
        if "2d_uvd" in preds:
            pred_uvd_joints = preds["2d_uvd"][:, : CONST.NUM_JOINTS, :].detach().cpu()
            inp_res = torch.Tensor(self.inp_res).cpu()
            pred_joints_2d = torch.einsum("bij,j->bij", pred_uvd_joints[:, :, :2], inp_res)
        else:
            pred_joints_2d = preds["joints_2d"].detach().cpu()

        joints_2d = targs[Queries.JOINTS_2D][:, : CONST.NUM_JOINTS, :].cpu()
        gt_root = joints_2d[:, 0, :]

        joints_vis = targs[Queries.JOINTS_VIS].cpu()

        # ==== draw pred and gt >>>>
        pred_image = self.draw_batch_joints_image(
            batch_images=image,
            batch_joints2d=pred_joints_2d,
            batch_gt_root2d=gt_root,
            batch_joints_vis=joints_vis,
        )
        ref_image = self.draw_batch_joints_image(
            batch_images=image,
            batch_joints2d=joints_2d,
            batch_gt_root2d=gt_root,
            batch_joints_vis=joints_vis,
        )
        self.image = np.concatenate([pred_image, ref_image], axis=1)
        self.count += batch_size