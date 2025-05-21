
import torch
from torch.cuda import is_available

def init_torch_device() -> torch.device:
	return torch.device("cuda" if is_available() else "cpu")
