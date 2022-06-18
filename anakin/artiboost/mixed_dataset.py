import torch
from anakin.utils.logger import logger


class MixedDataset(torch.utils.data.Dataset):

    def __init__(self, real_set, synth_set):
        super().__init__()
        self.real_set = real_set
        self.synth_set = synth_set

        self.real_len = len(self.real_set)
        self.synth_len = len(self.synth_set)

        self.epoch_len = self.real_len + self.synth_len

    def __len__(self):
        return self.epoch_len

    def remove_synth(self):
        # RandomDispathcer remove synth_set by shadow its index
        self.epoch_len = self.real_len

    def retrive_synth(self):
        self.epoch_len = self.real_len + self.synth_len

    def update(self):
        self.synth_len = len(self.synth_set)
        self.epoch_len = self.real_len + self.synth_len
        logger.info(f"MixedDataset has # real {self.real_len}, # synth {self.synth_len}")

    def __getitem__(self, index):
        if index >= self.real_len:
            return self.synth_set[index - self.real_len]
        else:
            sample = self.real_set[index]
            return sample
