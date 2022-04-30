import torch
import numpy as np


def load_data(path: str) -> None:
    print(f"loading data from {path}...")


def nparray_to_tensor(nparray, device: torch.device = torch.device("cpu")) -> torch.Tensor:
    temp = torch.from_numpy(nparray).float().to(device)
    return temp[None, :]
