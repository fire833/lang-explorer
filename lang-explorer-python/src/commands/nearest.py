
from sklearn.neighbors import NearestNeighbors
import pandas as pd
from src.utils.strlen import strlen
from graphviz import Source
from matplotlib import pyplot as plt

def nearest_neighbors(args):
	plain = args.input.removeprefix("results/")
	plain = plain.removesuffix(".csv")
	print("loading data...")
	data = pd.read_csv(args.input, skiprows=0, delimiter=',')

	psize = [len(str(x)) for x in data["type"]]
	data["plen"] = data["type"].apply(strlen)
	data = data.drop_duplicates()

	print(data.shape)

	indices = [int(i) for i in args.indices.split(",")]

	neigh = NearestNeighbors(n_neighbors=args.count, metric="euclidean")
	print("loading model...")
	neigh.fit(data.iloc[:,2:-1])

	query = data.iloc[indices, 2:-1]
	print(query.shape)

	(dist, neighind) = neigh.kneighbors(query)

	programs = data.iloc[indices, 0]
	graphs = data.iloc[indices, 1]

	fig, axes = plt.subplots(len(indices), len(indices) + 1, figsize=(7, 7))

	for i, prog in enumerate(programs):
		print(f"nearest neighbors for {prog}: ")

		file = f"/tmp/{i}_root.png"
		g = Source(graphs[indices[i]], file, format="png")

		axes[i][0].imshow(file)
		axes[i][0].axis('off')
		axes[i][0].set_title(f"Graph {i}")

		for j, neigh in enumerate(dist[i]):
			neighbor = data.iloc[neighind[i, j]]
			ngraph = neighbor[1]

			file = f"/tmp/{i}_{j}.png"
			img = Source(ngraph, file, format="png")

			axes[i][j + 1].imshow(file)
			axes[i][0].axis('off')
			axes[i][0].set_title(f"Graph {i}'s {j}th nearest neighbor")

			print(f"  {j + 1}: {neighbor[0]} {neigh}")

	plt.savefig("results/nearest_neighbors.png")
