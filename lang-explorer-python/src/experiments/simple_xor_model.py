#!/usr/bin/env python

from ray.train.torch import TorchTrainer, prepare_model
from ray.train import ScalingConfig
from torch.nn import Linear
from torch.nn import Module
from torch.nn import LeakyReLU
from torch.nn import MSELoss
from torch.optim import AdamW
import random
import torch

class XORModel(Module):
	def __init__(self):
		super(XORModel, self).__init__()

		self.layer1 = Linear(2, 15, bias=True)
		self.relu1 = LeakyReLU()
		self.layer2 = Linear(15, 1, bias=True)
		self.relu2 = LeakyReLU()

	def forward(self, x):
		x = self.layer1(x)
		x = self.relu1(x)
		x = self.layer2(x)
		x = self.relu2(x)
		return x

def ground_truth():
	x1 = random.randint(0, 255)
	x2 = random.randint(0, 255)

	return (torch.tensor([float(x1), float(x2)]), torch.tensor(float(x1 ^ x2)))

def train_func():
	model = XORModel()
	model = prepare_model(model)

	adam = AdamW(model.parameters(), lr=0.0001)
	loss = MSELoss()

def main():
	model = XORModel()

	model.train(mode=False)
	x, y = ground_truth()

	print(model.forward(x))

	# ray.init(f"ray://10.0.2.221:10001")

	# conf = ScalingConfig(num_workers=12, use_gpu=False)
	# trainer = TorchTrainer(scaling_config=conf)

if __name__ == "__main__":
	main()
