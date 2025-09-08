
from sklearn.neighbors import NearestNeighbors
import pandas as pd
from src.utils.strlen import strlen
from graphviz import Source
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import subprocess

def nearest_neighbors2(args):
	plain = args.input.removeprefix("results/")
	plain = plain.removesuffix(".csv")
	print("loading data...")
	data = pd.read_csv(args.input, skiprows=0, delimiter=',')

	psize = [len(str(x)) for x in data["type"]]
	data["plen"] = data["type"].apply(strlen)
	data = data.drop_duplicates()

	print(data.shape)

	indices = [int(i) for i in args.indices.split(",")]

	neigh = NearestNeighbors(n_neighbors=3, metric="euclidean")
	print("loading model...")
	neigh.fit(data.iloc[:,2:-1])

	query = data.iloc[indices, 2:-1]
	print(query.shape)

	(dist, neighind) = neigh.kneighbors(query)

	print(dist)
	print(neighind)

	programs = data.iloc[indices, 0]
	graphs = data.iloc[indices, 1]

	plt.title("Nearest neighbors examples")
	fig, axes = plt.subplots(2, 2)

	fig.tight_layout(pad=0.05)
	fig.subplots_adjust(top=0.99, bottom=0.01, left=0.01, right=0.99, hspace=0.001, wspace=0.001)

	path = Source(graphs[indices[0]]).render(f"/tmp/root", format="png", cleanup=True)

	axes[1, 0].imshow(mpimg.imread(path))
	axes[1, 0].axis('off')
	axes[1, 0].set_title(f"Original Program")

	# axes[i, 0].text(0.5, 0.001, prog, fontsize=1)
	print(f"Graph program: {programs}")

	for j, neigh in enumerate(dist[0]):
		if j == 0:
			continue
		
		x = 0
		y = 0
		if j == 1:
			x = 0
			y = 1
		if j == 2:
			x = 1
			y = 0
		if j == 3:
			x = 1
			y = 1

		neighprogram = data.iloc[neighind[0, j], 0]
		neighbor = data.iloc[neighind[0, j], 1]
		img = Source(neighbor).render(f"/tmp/{x}_{y}", format="png", cleanup=True)
		axes[x, y].imshow(mpimg.imread(img))
		axes[x, y].axis('off')
		axes[x, y].set_title(f"{j}th NN")
		# axes[i, j].text(0.5, 0.001, neighprogram, fontsize=1)


	plt.savefig(f"images/{args.output}.jpeg", dpi=1500)
	subprocess.run(["magick", f"images/{args.output}.jpeg", "-crop", "100%x50%", "+repage", "+adjoin", f"images/{args.output}_%d.jpeg"])
	subprocess.run(["mogrify", "-trim", f"images/{args.output}_*.jpeg"])
