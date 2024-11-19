import torch

print(torch.arange(10).sum())

print("CUDA", torch.cuda.is_available())