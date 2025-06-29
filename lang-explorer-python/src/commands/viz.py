
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os
import pandas as pd

def data_viz(args):
	for file in os.listdir("results/"):
		plain = file.removesuffix(".csv")
		data = pd.read_csv(f"results/{plain}.csv", skiprows=1, delimiter=',')

		embedding = TSNE(2).fit_transform(data.iloc[:,1:])
		plt.title(f"t-SNE (2D) of {plain} dataset")
		plt.scatter(embedding[:,0], embedding[:,1], c=data.iloc[:,-1])
		plt.savefig(f"{args.output}/tsne2d{plain}.png", dpi=500)
		plt.tight_layout()
		plt.close()

		embedding = TSNE(3).fit_transform(data.iloc[:,1:])
		fig = plt.figure(figsize=(8,6))
		ax = fig.add_subplot(111, projection='3d')
		ax.scatter3D(embedding[:,0], embedding[:,1], embedding[:,2], c=data.iloc[:,-1])
		ax.set_title(f"t-SNE (3D) of {plain} dataset")
		plt.savefig(f"{args.output}/tsne3d{plain}.png", dpi=500)
		plt.tight_layout()
		plt.close()
