
from sklearn.neighbors import NearestNeighbors
import pandas as pd
from src.utils.strlen import strlen
from graphviz import Source
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import subprocess

def nearest_neighbors3(args):
	programs = f"results/{args.lang}/{args.experiment_number}/graphviz.csv"
	nn = f"results/{args.lang}/{args.experiment_number}/embeddings_{args.embedding_system}_nn.csv"

	print("loading data...")
	progdata = pd.read_csv(args.input, skiprows=0, delimiter=',')
	nndata = pd.read_csv(nn, skiprows=0, delimiter=',')

	indices = [int(i) for i in args.indices.split(",")]

	neighbor_indices = nndata.iloc[indices, 0:4]
	neighbor_programs = progdata.iloc[neighbor_indices.values.flatten(), 1]

	plt.title("Nearest neighbors examples")
	fig, axes = plt.subplots(2, 2)

	fig.tight_layout(pad=0.05)
	fig.subplots_adjust(top=0.99, bottom=0.01, left=0.01, right=0.99, hspace=0.001, wspace=0.001)

	for j, neigh in enumerate(neighbor_programs):
		x = 0
		y = 0
		title = ""
		if j == 0:
			x = 0
			y = 0
			title = "Original Program"
		if j == 1:
			x = 0
			y = 1
			title = "1st NN"
		if j == 2:
			x = 1
			y = 0
			title = "2nd NN"
		if j == 3:
			x = 1
			y = 1
			title = "3rd NN"

		img = Source(neigh).render(f"/tmp/{x}_{y}", format="png", cleanup=True)
		axes[x, y].imshow(mpimg.imread(img))
		axes[x, y].axis('off')
		axes[x, y].set_title(title)
		# axes[i, j].text(0.5, 0.001, neighprogram, fontsize=1)

	plt.savefig(f"images/{args.output}.jpeg", dpi=1500)
	subprocess.run(["mogrify", "-trim", f"images/{args.output}.jpeg"])
