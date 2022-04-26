import torch


def load_data(path: str) -> None:
    print(f"loading data from {path}...")


def nparray_to_tensor(nparray, device: torch.device = torch.device("cpu")) -> torch.TensorType:
    temp = torch.from_numpy(nparray).float().to(device)
    return temp[None, :]


def concat_tensors(tensor1: torch.TensorType, tensor2: torch.TensorType) -> torch.TensorType:
    temp = torch.cat((tensor1[0], tensor1[0] - tensor2[0]))
    return temp[None, :]
