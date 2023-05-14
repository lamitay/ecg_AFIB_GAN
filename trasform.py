import torch

class Normalize(object):
    def __call__(self, tensor):
        return (tensor - torch.min(tensor)) / (torch.max(tensor) - torch.min(tensor))