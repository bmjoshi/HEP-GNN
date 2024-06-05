import torch
from torch_geometric.typing import OptTensor

kernel_list = ['gauss']
kernel_map = {}

def exponential_kernel(x: OptTensor):
    return torch.exp(-torch.abs(x))

def gaussian_kernel(x: OptTensor):
    return torch.exp(-x**2)

kernel_map['gauss'] = gaussian_kernel
kernel_map['exp']   = exponential_kernel