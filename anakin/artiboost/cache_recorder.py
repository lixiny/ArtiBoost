import atexit
import os
import pickle
import shutil
import signal
import sys

from anakin.utils.logger import logger


class CacheRecorder:

    def __init__(self, cache_root):
        self.cache_root = cache_root
        os.makedirs(cache_root, exist_ok=True)

        atexit.register(self.gracefully_quit, None, None)
        # handle ctrl-C and kill -9
        signal.signal(signal.SIGTERM, self.gracefully_quit)
        signal.signal(signal.SIGINT, self.gracefully_quit)

    def __call__(self, batch):
        # serialize data to cache_root
        for save_id, obj_id, persp_id, grasp_id, obj_name, final_obj_pose, final_hand_verts, final_joints in zip(
                batch["index"],
                batch["obj_id"],
                batch["persp_id"],
                batch["grasp_id"],
                batch["obj_name"],
                batch["final_obj_pose"],
                batch["final_hand_verts"],
                batch["final_joints"],
        ):
            save_path = os.path.join(self.cache_root, f"{int(save_id):0>4}.pkl")
            save_dict = {
                "obj_name": obj_name,
                "obj_id": obj_id.detach().cpu().item(),
                "persp_id": persp_id.detach().cpu().item(),
                "grasp_id": grasp_id.detach().cpu().item(),
                "obj_pose": final_obj_pose.detach().cpu().numpy(),
                "hand_verts": final_hand_verts.detach().cpu().numpy(),
                "hand_joints": final_joints.detach().cpu().numpy(),
            }
            with open(save_path, "wb") as stream:
                pickle.dump(save_dict, stream)

    def clear(self):
        if os.path.exists(self.cache_root):
            shutil.rmtree(self.cache_root)
        os.makedirs(self.cache_root, exist_ok=True)

    def gracefully_quit(self, sig_id, frame):
        if os.path.exists(self.cache_root):
            try:
                shutil.rmtree(self.cache_root)
                logger.info(f"clear manager's cache for render intermediates")
            except Exception:
                pass

        if sig_id:
            sys.exit()
