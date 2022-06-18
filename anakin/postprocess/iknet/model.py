import torch.nn as nn

from anakin.postprocess.iknet import utils


class IKNet(nn.Module):
    def __init__(
        self,
        njoints=21,
        hidden_size_pose=[256, 512, 1024, 1024, 512, 256],
    ):
        super(IKNet, self).__init__()
        self.njoints = njoints
        in_neurons = 3 * njoints
        out_neurons = 16 * 4  # 16 quats
        neurons = [in_neurons] + hidden_size_pose

        invk_layers = []
        for layer_idx, (inps, outs) in enumerate(zip(neurons[:-1], neurons[1:])):
            invk_layers.append(nn.Linear(inps, outs))
            invk_layers.append(nn.BatchNorm1d(outs))
            invk_layers.append(nn.ReLU())

        invk_layers.append(nn.Linear(neurons[-1], out_neurons))

        self.invk_layers = nn.Sequential(*invk_layers)

    def forward(self, joint):
        joint = joint.contiguous().view(-1, self.njoints * 3)
        quat = self.invk_layers(joint)
        quat = quat.view(-1, 16, 4)
        quat = utils.normalize_quaternion(quat)
        so3 = utils.quaternion_to_angle_axis(quat).contiguous()
        so3 = so3.view(-1, 16 * 3)
        return so3, quat