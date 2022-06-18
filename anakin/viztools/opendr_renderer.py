import numpy as np

from collections.abc import Iterable, Sized
from opendr.camera import ProjectPoints
from opendr.renderer import ColoredRenderer
from opendr.lighting import LambertianPointLight
import transforms3d as t3d
from .misc import CONSTANTS


class OpenDRRenderer(object):
    def __init__(self, img_size=256, flength=500.0):  # 822.79041):  #
        self.w = img_size
        self.h = img_size
        self.flength = flength

    def __call__(
        self,
        verts,
        faces,
        cam_intrinsics,
        img=None,
        do_alpha=False,
        far=None,
        near=None,
        vertex_color=None,
        img_size=None,
        R=None,
    ):
        """
        cam is 3D [fx, fy, px, py]
        """
        # deal with image size
        if img is not None:
            h, w = img.shape[:2]
        elif img_size is not None:
            h = img_size[0]
            w = img_size[1]
        else:
            h = self.h
            w = self.w

        # deal with verts and faces; if both np array, then ok
        # if both are lists / tuple, need to combine them and do offset
        # also vc, if appliciable
        if isinstance(verts, np.ndarray) and isinstance(faces, np.ndarray):
            # support only one mesh
            final_verts = verts
            final_faces = faces
            if vertex_color is not None:
                if vertex_color.ndim == 1:
                    final_vertex_color = np.repeat(vertex_color[None, ...], len(verts), axis=0)
                else:
                    final_vertex_color = vertex_color
            else:
                final_vertex_color = None
        elif (
            isinstance(verts, Iterable)
            and isinstance(verts, Sized)
            and isinstance(faces, Iterable)
            and isinstance(faces, Sized)
        ):
            # support multiple mesh
            assert len(verts) == len(faces), f"verts and faces do not match, got {len(verts)} and {len(faces)}"

            final_faces = []
            n_mesh = len(verts)
            curr_offset = 0  # offset of vert ids, alter faces
            for mesh_id in range(n_mesh):
                final_faces.append(faces[mesh_id] + curr_offset)
                curr_offset += len(verts[mesh_id])

            final_verts = np.concatenate(verts, axis=0)
            final_faces = np.concatenate(final_faces, axis=0)
            if vertex_color is not None:
                # it is tricky here, as we may need to repeat color
                # iterate and check
                # possible to optimize 2 loops into one if some one is willing to
                final_vertex_color = []
                for mesh_id in range(n_mesh):
                    if vertex_color[mesh_id].ndim == 1:
                        final_vertex_color.append(
                            np.repeat(vertex_color[mesh_id][None, ...], len(verts[mesh_id]), axis=0)
                        )
                    else:
                        final_vertex_color.append(vertex_color[mesh_id])
                final_vertex_color = np.concatenate(final_vertex_color, axis=0)
            else:
                final_vertex_color = None
        else:
            raise NotImplementedError(f"opendr do not support verts and faces, got type {type(verts)} and {type(faces)}")

        dist = np.zeros(5)
        dist = dist.flatten()
        M = np.eye(4)

        # get R, t from M (has to be world2cam)
        if R is None:
            R = M[:3, :3]
        ax, angle = t3d.axangles.mat2axangle(R)
        rt = ax * angle
        rt = rt.flatten()
        t = M[:3, 3]

        if cam_intrinsics is None:
            cam_intrinsics = np.array([[500, 0, 128], [0, 500, 128], [0, 0, 1]])

        pp = np.array([cam_intrinsics[0, 2], cam_intrinsics[1, 2]])
        f = np.array([cam_intrinsics[0, 0], cam_intrinsics[1, 1]])

        use_cam = ProjectPoints(
            rt=rt, t=t, f=f, c=pp, k=dist  # camera translation  # focal lengths  # camera center (principal point)
        )  # OpenCv distortion params

        if near is None:
            near = np.maximum(np.min(final_verts[:, 2]) - 25, 0.1)
        if far is None:
            far = np.maximum(np.max(final_verts[:, 2]) + 25, 25)

        imtmp = render_model(
            final_verts,
            final_faces,
            w,
            h,
            use_cam,
            do_alpha=do_alpha,
            img=img,
            far=far,
            near=near,
            color=final_vertex_color,
        )

        return (imtmp * 255).astype("uint8")


def simple_renderer(rn, verts, faces, yrot=np.radians(120), color=None):
    # Rendered model color
    rn.set(v=verts, f=faces, vc=color, bgcolor=np.ones(3))
    albedo = rn.vc

    # Construct Back Light (on back right corner)
    rn.vc = LambertianPointLight(
        f=rn.f,
        v=rn.v,
        num_verts=len(rn.v),
        light_pos=_rotateY(np.array([-200, -100, -100]), yrot),
        vc=albedo,
        light_color=np.array([1, 1, 1]),
    )

    # Construct Left Light
    rn.vc += LambertianPointLight(
        f=rn.f,
        v=rn.v,
        num_verts=len(rn.v),
        light_pos=_rotateY(np.array([800, 10, 300]), yrot),
        vc=albedo,
        light_color=np.array([1, 1, 1]),
    )

    # Construct Right Light
    rn.vc += LambertianPointLight(
        f=rn.f,
        v=rn.v,
        num_verts=len(rn.v),
        light_pos=_rotateY(np.array([-500, 500, 1000]), yrot),
        vc=albedo,
        light_color=np.array([0.7, 0.7, 0.7]),
    )

    return rn.r


def _create_renderer(w=640, h=480, rt=np.zeros(3), t=np.zeros(3), f=None, c=None, k=None, near=0.5, far=10.0):

    f = np.array([w, w]) / 2.0 if f is None else f
    c = np.array([w, h]) / 2.0 if c is None else c
    k = np.zeros(5) if k is None else k

    rn = ColoredRenderer()

    rn.camera = ProjectPoints(rt=rt, t=t, f=f, c=c, k=k)
    rn.frustum = {"near": near, "far": far, "height": h, "width": w}
    return rn


def _rotateY(points, angle):
    """Rotate the points by a specified angle."""
    ry = np.array([[np.cos(angle), 0.0, np.sin(angle)], [0.0, 1.0, 0.0], [-np.sin(angle), 0.0, np.cos(angle)]])
    return np.dot(points, ry)


def render_model(verts, faces, w, h, cam, near=0.5, far=25, img=None, do_alpha=False, color=None):
    rn = _create_renderer(w=w, h=h, near=near, far=far, rt=cam.rt, t=cam.t, f=cam.f, c=cam.c)

    # Uses img as background, otherwise white background.
    if img is not None:
        rn.background_image = img / 255.0 if img.max() > 1 else img

    if color is None:
        color = CONSTANTS.colors["light_blue"]

    imtmp = simple_renderer(rn, verts, faces, color=color)

    return imtmp