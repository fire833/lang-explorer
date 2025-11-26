
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import AgglomerativeClustering, SpectralClustering
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances, manhattan_distances 
import os
import pandas as pd
from scipy.cluster.hierarchy import dendrogram
from src.utils.strlen import strlen
import json

def data_viz2(args):
	lang = args.lang
	exp_number = args.experiment_number
	programs = f"results/{lang}/{exp_number}/programs.csv"
	gensimembed = f"results/{lang}/{exp_number}/embeddings_doc2vecgensim.csv"

	programs = pd.read_csv(programs)
	gensim = pd.read_csv(gensimembed)

	options = json.load(open(f"results/{lang}/{exp_number}/options.json", "r"))

	data = pd.merge(programs, gensim, on="idx", how="inner")

	psize = [len(str(x)) for x in data["program"]]
	data["plen"] = data["program"].apply(strlen)
	data = data.drop_duplicates()

	print(options)
	print(data.shape)
	print(data.head())

	embedding2 = TSNE(2).fit_transform(data.iloc[:,3:-1])
	plt.title(f"t-SNE (2D) of {lang} dataset")
	plt.scatter(embedding2[:,0], embedding2[:,1], c=data["plen"])
	plt.tight_layout()
	plt.savefig(f"{args.output}/tsne2d{lang}colors.svg", dpi=300)
	plt.close()

	plt.title(f"t-SNE (2D) of {lang} dataset")
	color_map = {True: "green", False: "blue"}
	label_map = {True: "Incomplete Program", False: "Complete Program"}
	colors = [color_map[val] for val in data["is_partial"]]
	# labels = [label_map[val] for val in data["is_partial"]]

	plt.scatter(embedding2[:,0], embedding2[:,1], c=colors)
	plt.tight_layout()
	plt.legend()
	plt.savefig(f"{args.output}/tsne2d{lang}ispartial.svg", dpi=300)
	plt.close()

	# embedding3 = TSNE(3).fit_transform(data.iloc[:,2:-1])
	# fig = plt.figure(figsize=(8,6))
	# ax = fig.add_subplot(111, projection='3d')
	# ax.scatter3D(embedding3[:,0], embedding3[:,1], embedding3[:,2], c=data["plen"])
	# ax.set_title(f"t-SNE (3D) of {plain} dataset")
	# plt.tight_layout()
	# plt.savefig(f"{args.output}/tsne3d{plain}colors.png", dpi=300)
	# plt.close()

	# clusters = AgglomerativeClustering(n_clusters=None, distance_threshold=0).fit(data.iloc[:,2:-1])
	# plt.title("Hierarchical Clustering Dendrogram")
	# # plot the top three levels of the dendrogram
	# plot_dendrogram(clusters, truncate_mode="level", p=3, leaf_rotation=90)
	# plt.xlabel("Number of points in node (or index of point if no parenthesis).")
	# plt.tight_layout()
	# plt.savefig(f"{args.output}/{plain}dendrogram.png", dpi=300)
	# plt.close()

	# plt.title(f"t-SNE (2D) of {plain} dataset (Hierachical clusters)")
	# plt.scatter(embedding2[:,0], embedding2[:,1], c=clusters.labels_)
	# plt.tight_layout()
	# plt.savefig(f"{args.output}/tsne2d{plain}colorshierachical.png", dpi=300)
	# plt.close()

	# fig = plt.figure(figsize=(8,6))
	# ax = fig.add_subplot(111, projection='3d')
	# ax.scatter3D(embedding3[:,0], embedding3[:,1], embedding3[:,2], c=clusters.labels_)
	# ax.set_title(f"t-SNE (3D) of {plain} dataset (Hierachical clusters)")
	# plt.tight_layout()
	# plt.savefig(f"{args.output}/tsne3d{plain}colorshierachical.png", dpi=300)
	# plt.close()

	# plot_similarity_scores(50, data, plain, args.output)

def plot_similarity_scores(count: int, data: pd.DataFrame, name: str, output: str):
	# vectors = data.sample(count, replace=False)
	vectors = data.iloc[5:count,0:-1]

	similarity = euclidean_distances(vectors.iloc[:,2:-1])

	plt.title(f"Euclidean distances between {count} vectors from {name}")
	plt.xticks(range(len(vectors)), labels=vectors["type"],
		rotation=75, fontsize=3, ha="right", rotation_mode="anchor")
	plt.yticks(range(len(vectors)), labels=vectors["type"], fontsize=3)
	plt.subplots_adjust(bottom=0.5)
	plt.imshow(similarity)
	plt.colorbar(shrink=0.4)
	plt.savefig(f"{output}/euclidean_similarity_{name}.jpeg", dpi=250)
	plt.close()

def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)
