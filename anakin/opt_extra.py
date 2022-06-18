import argparse

from anakin.opt import custom_arg_string


def data_generation_manager_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--opg_batch_size", type=int, default=256)
    parser.add_argument("--opg_num_workers", type=int, default=20)
    parser.add_argument("--gpu_render_id", type=str, required=True)

    parser.add_argument("--synth_root", type=str, default="/dev/shm/anakin")
    arg_extra, _ = parser.parse_known_args(custom_arg_string)

    ### temp
    arg_extra.ovg_batch_size = arg_extra.opg_batch_size
    arg_extra.ovg_num_workers = arg_extra.opg_num_workers
    return arg_extra
