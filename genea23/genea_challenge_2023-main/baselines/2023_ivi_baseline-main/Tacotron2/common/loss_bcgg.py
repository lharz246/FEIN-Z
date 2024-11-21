from torch import nn
import torch
from torch.nn import functional as F

class BCGLoss(nn.Module):
    def __init__(self, weights):
        super(BCGLoss, self).__init__()
        self.joints_weights = weights['joints']
        self.velocity_weights = weights['velocity']
        self.acceleration_weights = weights['acceleration']

    def forward(self, predicted, ground_truth):
        # positional loss
        # prepend_size = predicted.shape
        ground_truth = ground_truth.transpose(1, 2)
        ground_truth.requires_grad = False
        loss_joint_positions = F.smooth_l1_loss(predicted / 0.1, ground_truth / 0.1) * 0.1

        # velocity loss
        velocity_predicted = torch.diff(predicted, n=1, dim=1)  # , prepend=torch.zeros(prepend_size).cuda())
        velocity_ground_truth = torch.diff(ground_truth, n=1, dim=1)  # , prepend=torch.zeros(prepend_size).cuda())
        velocity_ground_truth.requires_grad = False
        loss_velocity = F.smooth_l1_loss(velocity_predicted / 0.1, velocity_ground_truth / 0.1) * 0.1

        # acceleration loss
        acceleration_predicted = torch.diff(velocity_predicted, n=1, dim=1)  # , prepend=torch.zeros(prepend_size).cuda())
        acceleration_ground_truth = torch.diff(velocity_ground_truth, n=1, dim=1)  # ,
        # prepend=torch.zeros(prepend_size).cuda())
        acceleration_ground_truth.requires_grad = False
        loss_acceleration = F.smooth_l1_loss(acceleration_predicted / 0.1, acceleration_ground_truth / 0.1) * 0.1
        loss = loss_joint_positions * self.joints_weights + loss_velocity * self.velocity_weights + \
               loss_acceleration * self.acceleration_weights
        # maybe smth for embedding

        return loss
