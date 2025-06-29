
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import AgglomerativeClustering, SpectralClustering
import os
import pandas as pd
from scipy.cluster.hierarchy import dendrogram

def strlen(x):
	return len(str(x))

def data_viz(args):
	for file in os.listdir("results/"):
		plain = file.removesuffix(".csv")
		if args.input != "" and plain != args.input:
			continue
 
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

		clusters = AgglomerativeClustering(n_clusters=None, distance_threshold=0).fit(data.iloc[:,1:-1])
		plt.title("Hierarchical Clustering Dendrogram")
		# plot the top three levels of the dendrogram
		plot_dendrogram(clusters, truncate_mode="level", p=3, leaf_rotation=90)
		plt.xlabel("Number of points in node (or index of point if no parenthesis).")
		plt.tight_layout()
		plt.savefig(f"{args.output}/{plain}dendrogram.png", dpi=500)
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
