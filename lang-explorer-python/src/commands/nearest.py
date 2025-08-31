
from sklearn.neighbors import NearestNeighbors
import pandas as pd
from src.utils.strlen import strlen
from graphviz import Source
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import subprocess

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

	neigh = NearestNeighbors(n_neighbors=args.count + 1, metric="euclidean")
	print("loading model...")
	neigh.fit(data.iloc[:,2:-1])

	query = data.iloc[indices, 2:-1]
	print(query.shape)

	(dist, neighind) = neigh.kneighbors(query)

	programs = data.iloc[indices, 0]
	graphs = data.iloc[indices, 1]

	print(programs)

	plt.title("Nearest neighbors examples")
	fig, axes = plt.subplots(len(indices), args.count + 1)

	fig.tight_layout(pad=0.05)
	fig.subplots_adjust(top=0.99, bottom=0.01, left=0.01, right=0.99, hspace=0.001, wspace=0.001)

	for i, prog in enumerate(programs):
		path = Source(graphs[indices[i]]).render(f"/tmp/{i}_root", format="png", cleanup=True)

		axes[i, 0].imshow(mpimg.imread(path))
		axes[i, 0].axis('off')
		axes[i, 0].set_title(f"Graph {i}")

		for j, neigh in enumerate(dist[i]):
			if j == 0:
				continue

			neighbor = data.iloc[neighind[i, j], 1]

			img = Source(neighbor).render(f"/tmp/{i}_{j}", format="png", cleanup=True)

			axes[i, j].imshow(mpimg.imread(img))
			axes[i, j].axis('off')
			axes[i, j].set_title(f"{j}th NN")

	plt.savefig(f"images/{args.output}.jpeg", dpi=1500)
	subprocess.run(["magick", f"images/{args.output}.jpeg", "-crop", "100%x50%", "+repage", "+adjoin", f"images/{args.output}_%d.jpeg"])
	subprocess.run(["mogrify", "-trim", f"images/{args.output}_*.jpeg"])
