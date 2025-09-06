
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from src.utils.client import generate, TacoExpressionParameters, TacoScheduleParameters, CSSLanguageParameters, GenerateParams, GenerateResults, GeneralTrainingParameters, KarelLanguageParameters
import os
import pandas as pd

def generate_embeddings(args):
	embeddings = ["mxbailarge"]

	print("making call to explorer")
	res = generate("http://localhost:8080", args.language, "wmc",
		GenerateParams(args.count, False, True, True, False, True, embeddings, args.wl_count, args.num_neg_samples, 128, 3, 3, args.grad_clip, "Average", GeneralTrainingParameters(512, args.epochs, args.learning_rate, 0.9, args.seed, 50, False, True),
		css=CSSLanguageParameters("exhaustivev1", ["div", "h1", "h2", "h3", "h4", "h5", "h6", "a"], ["foobar"], [
			"#842d5b",
	        "#20b01c",
	        "#7d1dc1",
	        "#42a1dc",
	        "#da8454",
	        "#8ec5d2",
	        "#a69657",
	        "#a69657",
	        "#664ba3",
	        "#3b6a42",
	        "rgb(39, 37, 193)",
	        "rgb(37, 138, 166)",
	        "rgb(84, 183, 126)",
	        "rgb(104, 36, 170)",
	        "rgb(207, 106, 144)",
	        "rgb(203, 135, 198)",
	        "rgb(231, 100, 187)",
	        "rgb(143, 143, 143)",
	        "rgb(68, 68, 68)",
	        "rgb(160, 160, 160)",
	        "rgb(141, 141, 141)",
	        "rgb(95, 95, 95)",
		]),
		taco_expression=TacoExpressionParameters(args.version, 
	        ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"], 
	        ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"]),
		taco_schedule=TacoScheduleParameters(args.version, index_variables=["a", "b", "c", "d", "e"], workspace_index_variables=["f", "g", "h", "i"], fused_index_variables=["j", "k", "l", "m"], split_factor_variables=["n", "o", "p", "q"], divide_factor_variables=["r", "s", "t", "u"], unroll_factor_variables=["v", "w", "x", "y", "z"]),
	    karel=KarelLanguageParameters()))	
	document_collections = []

	print("extracting explorer response")
	for i, v in enumerate(res.programs):
		graph = v["program"]
		# print(v["graphviz"])
		doc = TaggedDocument(words=v["features"], tags=[graph])
		document_collections.append(doc)

	print("\nOptimization started.\n")

	model = Doc2Vec(document_collections,
		vector_size=args.dimensions,
		window=0,
		min_count=args.min_count,
		dm=0,
		sample=args.down_sampling,
		workers=args.workers,
		epochs=args.epochs,
		compute_loss=True,
		alpha=args.learning_rate)

	# print(f"{model.running_training_loss}, {model.cum_table}, {model.compute_loss}")

	output1 = f"results/embeddings_{args.dimensions}_{args.epochs}_{args.count}_{args.language}.csv"
	save_embedding(output1, model, res.programs, args.dimensions)

	for (name, _) in embeddings:
		output = f"results/embeddings_{args.dimensions}_{args.count}_{args.language}_{name.replace("-", "_")}.csv"
		save_embedding_new(output, res.programs, name, args.dimensions)

def save_embedding_new(output_path, programs, type, dimensions):
	"""
	Function to save the embedding.
	:param output_path: Path to the embedding csv.
	:param programs: The list of program results.
	:param dimensions: The embedding dimension parameter.
	"""
	out = []
	for prog in programs:
		out.append([prog["program"], prog["graphviz"]] + list(prog["embeddings"][type]))
	column_names = ["type", "graphviz"] + ["x_" + str(dim) for dim in range(dimensions)]
	out = pd.DataFrame(out, columns=column_names)
	out = out.sort_values(["type"])
	out.to_csv(output_path, index=None)

def save_embedding(output_path, model, programs, dimensions):
	"""
	Function to save the embedding.
	:param output_path: Path to the embedding csv.
	:param model: The embedding model object.
	:param files: The list of files.
	:param dimensions: The embedding dimension parameter.
	"""
	out = []
	for prog in programs:
		out.append([prog["program"], prog["graphviz"]] + list(model.docvecs[prog["program"]])) 
	column_names = ["type" ,"graphviz"] + ["x_"+str(dim) for dim in range(dimensions)]
	out = pd.DataFrame(out, columns=column_names)
	out = out.sort_values(["type"])
	out.to_csv(output_path, index=None)
