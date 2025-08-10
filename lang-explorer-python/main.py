#!/usr/bin/env python

from src.utils.client import GenerateParams, GenerateResults, CSSLanguageParameters, TacoScheduleParameters, TacoExpressionParameters
from argparse import ArgumentParser	
import sys
from src.commands.viz import data_viz
from src.commands.embeddings import generate_embeddings
from src.commands.interact import interactive_tsne
from src.commands.nearest import nearest_neighbors
import matplotlib

# %matplotlib qt

def main():
	parser = ArgumentParser(description="Lang-Explorer Python stuff")
	sub = parser.add_subparsers(dest="cmd", required=True)

	embed = sub.add_parser("embedgen", help="Generate embeddings.")
	embed.add_argument("--dimensions", type=int, default=128, help="Number of dimensions. Default is 128.")
	embed.add_argument("--epochs", type=int, default=10, help="Number of epochs. Default is 10.")
	embed.add_argument("--min-count", type=int, default=5, help="Minimal structural feature count. Default is 5.")
	embed.add_argument("--learning-rate", type=float, default=0.025, help="Initial learning rate. Default is 0.025.")
	embed.add_argument("--down-sampling", type=float, default=0.0001, help="Down sampling rate of features. Default is 0.0001.")
	embed.add_argument("--workers", type=int, default=8, help="Number of workers. Default is 8.")
	embed.add_argument("--language", type=str, default="", help="Specify the language to generate embeddings with.")
	embed.add_argument("--version", type=str, default="contextfreev1", help="Specify the version of the language to generate embeddings with")
	embed.add_argument("--count", type=int, default=10000, help="Specify the number of samples to retrieve.")
	embed.add_argument("--wl-count", type=int, default=3, help="Specify the number of WL kernel iterations to run.")
	embed.add_argument("--num_neg_samples", type=int, default=64, help="Specify number of negative samples to update.")
	embed.add_argument("--batch_size", type=int, default=128, help="Specify the batch size to train on.")
	embed.add_argument("--seed", type=int, default=50, help="Specify a seed when generating and training data.")
	embed.set_defaults(func=generate_embeddings)

	viz = sub.add_parser("dataviz", help="Visualize embedding spaces.")
	viz.add_argument("--output", nargs="?", default="images", help="Images path.")
	viz.add_argument("--input", default="embeddings.csv", help="Specify the embeddings to use for visualization.")
	viz.set_defaults(func=data_viz)

	interactive = sub.add_parser("dataint", help="Interact with TSNE data.")
	interactive.add_argument("--input", default="embeddings.csv", help="Specify the embeddings to use for visualization.")
	interactive.set_defaults(func=interactive_tsne)

	neighbors = sub.add_parser("neigh", help="Get nearest neighbors for a particular set of neighbors")
	neighbors.add_argument("--input", default="embeddings.csv", help="Specify the embeddings to use for visualization.")
	neighbors.add_argument("--indices", help="Comma separated list of indices that you want the nearest neighbors of.")
	neighbors.add_argument("--count", type=int, default=10, help="Number of nearest neighbors to retrieve.")
	neighbors.set_defaults(func=nearest_neighbors)

	args = parser.parse_args(sys.argv[1:])
	args.func(args)

if __name__ == "__main__":
	main()
