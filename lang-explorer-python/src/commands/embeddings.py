
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from src.utils.client import generate, TacoExpressionParameters, TacoScheduleParameters, CSSLanguageParameters, GenerateParams, GenerateResults
import os
import pandas as pd

def generate_embeddings(args):
	print("making call to explorer")
	res = generate("http://localhost:8080", "tacosched", "mc", 
		GenerateParams(100000, True, True, False, 3, 
		css=CSSLanguageParameters(["foo", "bar", "baz"], ["1", "2", "3"]),
		# taco_expression=TacoExpressionParameters(),
		# taco_schedule=TacoScheduleParameters(),
		taco_schedule=TacoScheduleParameters(index_variables=["i"], workspace_index_variables=["j"], fused_index_variables=["k"], split_factor_variables=["l"], divide_factor_variables=["m"], unroll_factor_variables=["n"]),
		taco_expression=TacoExpressionParameters([], [])))

	document_collections = []

	print("extracting explorer response")
	for i, feat in enumerate(res.features):
		graph = res.programs[i]
		doc = TaggedDocument(words=feat, tags=[graph])
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
		alpha=args.learning_rate)

	save_embedding(args.output_path, model, res.programs, args.dimensions)

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
        out.append([prog] + list(model.docvecs[prog]))
    column_names = ["type"]+["x_"+str(dim) for dim in range(dimensions)]
    out = pd.DataFrame(out, columns=column_names)
    out = out.sort_values(["type"])
    out.to_csv(output_path, index=None)
