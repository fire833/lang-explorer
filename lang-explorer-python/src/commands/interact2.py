
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

def interactive_tsne2(args):
	lang = args.lang
	exp_number = args.experiment_number
	programs = f"results/{lang}/{exp_number}/programs.csv"
	gensimembed = f"results/{lang}/{exp_number}/embeddings_doc2vecgensim.csv"

	programs = pd.read_csv(programs)
	gensim = pd.read_csv(gensimembed)

	data = pd.merge(programs, gensim, on="idx", how="inner")

	psize = [len(str(x)) for x in data["program"]]
	data["plen"] = data["program"].apply(strlen)
	data = data.drop_duplicates()

	print(data.shape)
	print(data.head())

	embedding2 = TSNE(2).fit_transform(data.iloc[:,3:-1])

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
	plt.title(f"t-SNE (2D) of {lang} dataset")
	plt.show()
