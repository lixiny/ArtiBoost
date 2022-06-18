from enum import Enum, auto

from anakin.utils.misc import ImmutableClass


class Queries(metaclass=ImmutableClass):
    SAMPLE_IDX = "sample_idx"
    RAW_IMAGE = "raw_image"
    IMAGE = "image"
    IMAGE_PATH = "image_path"
    CAM_INTR = "cam_intr"
    ORTHO_INTR = "ortho_intr"

    OBJ_VERTS_CAN = "obj_verts_can"
    OBJ_VERTS_3D = "obj_verts_3d"
    OBJ_VERTS_2D = "obj_verts_2d"
    HAND_VERTS_3D = "hand_verts_3d"
    HAND_VERTS_2D = "hand_verts_2d"

    CORNERS_CAN = "corners_can"
    CORNERS_2D = "corners_2d"
    CORNERS_3D = "corners_3d"
    JOINTS_2D = "joints_2d"
    JOINTS_3D = "joints_3d"
    ROOT_JOINT = "root_joint"
    BONE_SCALE = "bone_scale"

    JOINTS_HEATMAP = "joints_heatmap"
    CORNERS_HEATMAP = "corners_heatmap"

    CORNERS_VIS = "corners_vis"
    JOINTS_VIS = "joints_vis"

    OBJ_TRANSF = "obj_transf"
    OBJ_FACES = "obj_faces"
    HAND_SHAPE = "hand_shape"
    HAND_POSE = "hand_pose"  # deprecated in the future
    HAND_FACES = "hand_faces"

    BBOX_CENTER = "bbox_center"
    BBOX_SCALE = "bbox_scale"

    HAND_BBOX = "hand_bbox"

    OBJ_IDX = "obj_idx"

    SIDE = "side"
    PADDING_MASK = "padding_mask"
    FACE_PADDING_MASK = "face_padding_mask"


class SynthQueries(metaclass=ImmutableClass):
    IS_SYNTH = "is_synth"
    OBJ_ID = "obj_id"
    PERSP_ID = "persp_id"
    GRASP_ID = "grasp_id"


def match_collate_queries(query_spin):
    object_vertex_queries = [
        Queries.OBJ_VERTS_3D,
        Queries.OBJ_VERTS_CAN,
        Queries.OBJ_VERTS_2D,
    ]
    object_face_quries = [
        Queries.OBJ_FACES,
    ]

    if query_spin in object_vertex_queries:
        return Queries.PADDING_MASK
    elif query_spin in object_face_quries:
        return Queries.FACE_PADDING_MASK
