import torch.nn as nn


def vgg16_loss(outputs, target):
    cross_entropy_loss = nn.CrossEntropyLoss()
    loss = cross_entropy_loss(outputs, target)
    return loss
