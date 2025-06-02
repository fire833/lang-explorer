#!/usr/bin/env python

from ray.train.torch import TorchTrainer, prepare_model
from ray.train import ScalingConfig
from torch.nn import Linear
from torch.nn import Module
from torch.nn import LeakyReLU
from torch.nn import MSELoss
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from lightning.pytorch import LightningModule, Trainer
import torch.nn.functional as F
import random
import torch
import math

class XORDataset(Dataset):
	def __init__(self, size: int):
		self.data = [truth(i/1000) for i in range(-size, size, 1)]

	def __len__(self):
		return self.data.__len__()
	
	def __getitem__(self, idx: int):
		return self.data[idx]

class XORModel(Module):
	def __init__(self):
		super(XORModel, self).__init__()

		self.layer1 = Linear(1, 20, bias=True)
		self.layer3 = Linear(20, 1, bias=True)

	def forward(self, x):
		x = self.layer1(x)
		x = F.relu(x)
		x = self.layer3(x)
		x = F.relu(x)
		return x

class XORLearner(LightningModule):
	def __init__(self):
		super(XORLearner, self).__init__()
		self.model = XORModel()

	def forward(self, x):
		return self.model.forward(x)
	
	def configure_optimizers(self):
		return Adam(self.parameters(), lr=0.01)

	def training_step(self, batch, batch_idx):
		(x, y) = batch
		yhat = self.model.forward(x)
		loss = F.mse_loss(yhat, y)
		self.log("train_loss", loss, on_epoch=True, prog_bar=True)
		return loss

	def validation_step(self, batch, batch_idx):
		(x, y) = batch
		yhat = self.model.forward(x)
		loss = F.mse_loss(yhat, y)
		self.log("validation_loss", loss, on_epoch=True, prog_bar=True)
		return loss

def truth(x):
	return (torch.tensor(x), torch.tensor(x**3))

def train_func():
	model = XORModel()
	model = prepare_model(model)

	adam = Adam(model.parameters(), lr=0.01)
	loss = MSELoss()

def main():
	print("building datasets")
	train = DataLoader(XORDataset(1000), batch_size=1, shuffle=True, num_workers=2)
	validate = DataLoader(XORDataset(500), batch_size=1, shuffle=False, num_workers=2)

	print("instantiating trainer")
	trainer = Trainer(log_every_n_steps=5)
	print("training...")
	model = XORLearner()
	trainer.fit(model, train, validate)

	# ray.init(f"ray://10.0.2.221:10001")

	# conf = ScalingConfig(num_workers=12, use_gpu=False)
	# trainer = TorchTrainer(scaling_config=conf)

if __name__ == "__main__":
	main()
