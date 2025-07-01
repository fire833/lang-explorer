
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.manifold import TSNE
from src.utils.strlen import strlen
import mplcursors

def split_string(item: str, items: list) -> list:
	cap = 75

	if len(item) > cap:
		line = item[0:cap]
		items.append(line)
		rem = item[cap:]
		return split_string(rem, items)
	else:
		items.append(item)
		return items

def interactive_tsne(args):
	plain = args.input.removeprefix("results/")
	plain = plain.removesuffix(".csv")
	data = pd.read_csv(args.input, skiprows=0, delimiter=',')

	psize = [len(str(x)) for x in data["type"]]
	data["plen"] = data["type"].apply(strlen)
	data = data.drop_duplicates()

	print(data.shape)

	embedding2 = TSNE(2).fit_transform(data.iloc[:,1:-1])

	fig, ax = plt.subplots()
	sc = ax.scatter(embedding2[:,0], embedding2[:,1], c=data["plen"])
	cursor = mplcursors.cursor(sc, hover=True)

	@cursor.connect("add")
	def on_add(sel):
		label = data.iloc[sel.index, 0]
		labels = split_string(label, [])
		sel.annotation.set_text("\n".join(labels))

	# print(f"creating {data.__len__()} annotations...")
	# for i, _ in enumerate(data):
	# 	ax.annotate(data.iloc[i, 0], (embedding2[i, 0], embedding2[i, 1]))

	# Connect the event
	plt.title(f"t-SNE (2D) of {plain} dataset")
	plt.show()
