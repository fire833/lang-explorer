#!/usr/bin/env python

from ray.train.torch import TorchTrainer, prepare_model
from ray.train import ScalingConfig
from ray.train.lightning import prepare_trainer
from torch.nn import Linear
from torch.nn import Module
from torch.nn import LeakyReLU
from torch.nn import MSELoss
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from lightning.pytorch import LightningModule, Trainer
import lightning as pl
import torch.nn.functional as F
import random
import torch
import math
import ray

class CubeDataset(Dataset):
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
	print("loading datasets...")
	train = DataLoader(CubeDataset(1000), batch_size=1, shuffle=True, num_workers=2)
	validate = DataLoader(CubeDataset(500), batch_size=1, shuffle=False, num_workers=2)

	model = XORLearner()
	print("configuring trainer...")
    # [1] Configure PyTorch Lightning Trainer.
	trainer = pl.Trainer(
		max_epochs=30,
		devices="auto",
		accelerator="auto",
		strategy=ray.train.lightning.RayDDPStrategy(),
		plugins=[ray.train.lightning.RayLightningEnvironment()],
		callbacks=[ray.train.lightning.RayTrainReportCallback()],
		# [1a] Optionally, disable the default checkpointing behavior
		# in favor of the `RayTrainReportCallback` above.
		enable_checkpointing=False,
    )

	trainer = ray.train.lightning.prepare_trainer(trainer)
	trainer.fit(model, train_dataloaders=train, val_dataloaders=validate)

def main():
	ray.init(f"ray://10.0.2.221:10001", runtime_env={
		"pip": ["torch", "lightning"],
	})
	scaling_config = ScalingConfig(num_workers=10, use_gpu=True)

	# [3] Launch distributed training job.
	trainer = TorchTrainer(
    	train_func,
    	scaling_config=scaling_config,
    	# [3a] If running in a multi-node cluster, this is where you
    	# should configure the run's persistent storage that is accessible
    	# across all worker nodes.
    	# run_config=ray.train.RunConfig(storage_path="s3://..."),
	)
	result: ray.train.Result = trainer.fit()


if __name__ == "__main__":
	main()
