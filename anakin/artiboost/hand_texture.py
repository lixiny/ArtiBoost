import trimesh
from typing import List


class HTMLHand:
    @classmethod
    def get_HTML_mesh(cls) -> List[trimesh.base.Trimesh]:
        hand_mesh = [
            trimesh.load(f"data/HTML_supp/html_{i + 1:03d}/hand.obj", process=False) for i in range(52) if i != 2
        ]
        return hand_mesh
