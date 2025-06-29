
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os
import pandas as pd

def strlen(x):
	return len(str(x))

def data_viz(args):
	for file in os.listdir("results/"):
		plain = file.removesuffix(".csv")
		data = pd.read_csv(f"results/{plain}.csv", skiprows=0, delimiter=',')

		psize = [len(str(x)) for x in data["type"]]
		data["plen"] = data["type"].apply(strlen)
		data = data.drop_duplicates()

		print(data.shape)

		embedding = TSNE(2).fit_transform(data.iloc[:,1:-1])
		plt.title(f"t-SNE (2D) of {plain} dataset")
		plt.scatter(embedding[:,0], embedding[:,1], c=data["plen"])
		plt.savefig(f"{args.output}/tsne2d{plain}colors.png", dpi=500)
		plt.tight_layout()
		plt.legend()
		plt.close()

		embedding = TSNE(3).fit_transform(data.iloc[:,1:-1])
		fig = plt.figure(figsize=(8,6))
		ax = fig.add_subplot(111, projection='3d')
		ax.scatter3D(embedding[:,0], embedding[:,1], embedding[:,2], c=data["plen"])
		ax.set_title(f"t-SNE (3D) of {plain} dataset")
		plt.savefig(f"{args.output}/tsne3d{plain}colors.png", dpi=500)
		plt.tight_layout()
		plt.legend()
		plt.close()
