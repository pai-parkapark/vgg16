import torch


def vgg16_accuracy(output, target):
    total = 0
    correct = 0

    _, predicted = torch.max(output.data, 1)
    total += target.size(0)
    correct += (predicted == target).sum().item()

    return total, correct
