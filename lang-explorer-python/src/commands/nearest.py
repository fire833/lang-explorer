
from sklearn.neighbors import NearestNeighbors
import pandas as pd
from src.utils.strlen import strlen

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
	neigh.fit(data.iloc[:,1:-1])

	query = data.iloc[indices, 1:-1]
	print(query.shape)

	(dist, neighind) = neigh.kneighbors(query)

	programs = data.iloc[indices, 0]

	for i, prog in enumerate(programs):
		print(f"nearest neighbors for {prog}: ")
		for j, neigh in enumerate(dist[i]):
			print(f"  {j + 1}: {data.iloc[neighind[i, j], 0]} {neigh}")
