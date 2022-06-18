import copy
import os

import numpy as np
import trimesh
from pyrender import Mesh, OffscreenRenderer, Primitive, Renderer, Scene
from pyrender.constants import GLTF, RenderFlags
from pyrender.light import DirectionalLight, PointLight, SpotLight
from pyrender.material import MetallicRoughnessMaterial


class FScene(Scene):
    def get_node(self, name):
        return super().get_nodes(name=name).pop()

    def show_node(self, name, pose=None):
        node = self.get_node(name)
        node.mesh.is_visible = True
        if pose is not None:
            node.matrix = pose

    def hide_node(self, name):
        node = self.get_node(name)
        node.mesh.is_visible = False


class FPrimitive(Primitive):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._update_positions = False

    @property
    def positions(self):
        return super().positions

    @positions.setter
    def positions(self, value):
        Primitive.positions.fset(self, value)
        # super().positions = value
        self._update_positions = True

    def _update_to_context(self):
        if self._update_positions:
            self._remove_from_context()
            self._add_to_context()
            self._update_positions = False


class FRenderer(Renderer):
    def _update_context(self, scene, flags):

        # Update meshes
        scene_meshes = scene.meshes

        # Add new meshes to context
        for mesh in scene_meshes - self._meshes:
            for p in mesh.primitives:
                p._add_to_context()

        # Remove old meshes from context
        for mesh in self._meshes - scene_meshes:
            for p in mesh.primitives:
                p.delete()

        # ? >>>>>> update context directly >>>>>
        for mesh in self._meshes & scene_meshes:
            for p in mesh.primitives:
                p._update_to_context()
        # ? <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        self._meshes = scene_meshes.copy()

        # Update mesh textures
        mesh_textures = set()
        for m in scene_meshes:
            for p in m.primitives:
                mesh_textures |= p.material.textures

        # Add new textures to context
        for texture in mesh_textures - self._mesh_textures:
            texture._add_to_context()

        # Remove old textures from context
        for texture in self._mesh_textures - mesh_textures:
            texture.delete()

        self._mesh_textures = mesh_textures.copy()

        shadow_textures = set()
        for l in scene.lights:
            # Create if needed
            active = False
            if isinstance(l, DirectionalLight) and flags & RenderFlags.SHADOWS_DIRECTIONAL:
                active = True
            elif isinstance(l, PointLight) and flags & RenderFlags.SHADOWS_POINT:
                active = True
            elif isinstance(l, SpotLight) and flags & RenderFlags.SHADOWS_SPOT:
                active = True

            if active and l.shadow_texture is None:
                l._generate_shadow_texture()
            if l.shadow_texture is not None:
                shadow_textures.add(l.shadow_texture)

        # Add new textures to context
        for texture in shadow_textures - self._shadow_textures:
            texture._add_to_context()

        # Remove old textures from context
        for texture in self._shadow_textures - shadow_textures:
            texture.delete()

        self._shadow_textures = shadow_textures.copy()


class FMesh(Mesh):
    def update_verts(self, positions: np.ndarray, pid: int = 0):
        self.primitives[pid].positions = positions

    @staticmethod
    def from_trimesh(mesh, material=None, is_visible=True, poses=None, wireframe=False, smooth=True):

        if isinstance(mesh, (list, tuple, set, np.ndarray)):
            meshes = list(mesh)
        elif isinstance(mesh, trimesh.Trimesh):
            meshes = [mesh]
        else:
            raise TypeError("Expected a Trimesh or a list, got a {}".format(type(mesh)))

        primitives = []
        for m in meshes:
            positions = None
            normals = None
            indices = None

            # Compute positions, normals, and indices
            if smooth:
                positions = m.vertices.copy()
                normals = m.vertex_normals.copy()
                indices = m.faces.copy()
            else:
                positions = m.vertices[m.faces].reshape((3 * len(m.faces), 3))
                normals = np.repeat(m.face_normals, 3, axis=0)

            # Compute colors, texture coords, and material properties
            color_0, texcoord_0, primitive_material = Mesh._get_trimesh_props(m, smooth=smooth, material=material)

            # Override if material is given.
            if material is not None:
                # primitive_material = copy.copy(material)
                primitive_material = copy.deepcopy(material)

            if primitive_material is None:
                # Replace material with default if needed
                primitive_material = MetallicRoughnessMaterial(
                    alphaMode="BLEND", baseColorFactor=[0.3, 0.3, 0.3, 1.0], metallicFactor=0.2, roughnessFactor=0.8
                )

            primitive_material.wireframe = wireframe

            # Create the primitive
            # ? Use our Fast Primitive
            primitives.append(
                FPrimitive(
                    positions=positions,
                    normals=normals,
                    texcoord_0=texcoord_0,
                    color_0=color_0,
                    indices=indices,
                    material=primitive_material,
                    mode=GLTF.TRIANGLES,
                    poses=poses,
                )
            )

        return FMesh(primitives=primitives, is_visible=is_visible)


class FOffscreenRenderer(OffscreenRenderer):
    def __init__(self, viewport_width, viewport_height, point_size, gpu_id: int = 0):
        self.viewport_width = viewport_width
        self.viewport_height = viewport_height
        self.point_size = point_size

        self._platform = None
        self._renderer = None
        self._create_with_id(gpu_id)

    def _create_with_id(self, gpu_id: int = 0):
        if "PYOPENGL_PLATFORM" not in os.environ:
            raise NotImplementedError
        elif os.environ["PYOPENGL_PLATFORM"] == "egl":
            from pyrender.platforms import egl

            # os.environ.get("EGL_DEVICE_ID", "0")
            device_id = gpu_id
            egl_device = egl.get_device_by_index(device_id)
            self._platform = egl.EGLPlatform(self.viewport_width, self.viewport_height, device=egl_device)
        elif os.environ["PYOPENGL_PLATFORM"] == "osmesa":
            raise NotImplementedError
        else:
            raise ValueError("Unsupported PyOpenGL platform: {}".format(os.environ["PYOPENGL_PLATFORM"]))
        self._platform.init_context()
        self._platform.make_current()
        self._renderer = FRenderer(self.viewport_width, self.viewport_height)
