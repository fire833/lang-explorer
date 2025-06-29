#!/usr/bin/env python

from src.utils.client import GenerateParams, GenerateResults, CSSLanguageParameters, TacoScheduleParameters, TacoExpressionParameters
from argparse import ArgumentParser	
import sys
from src.commands.viz import data_viz
from src.commands.embeddings import generate_embeddings

def main():
	parser = ArgumentParser(description="Lang-Explorer Python stuff")
	sub = parser.add_subparsers(dest="cmd", required=True)

	embed = sub.add_parser("embedding_gen", help="Generate embeddings.")
	embed.add_argument("--output-path", nargs="?", default="results/embeddings.csv", help="Embeddings path.")
	embed.add_argument("--dimensions", type=int, default=128, help="Number of dimensions. Default is 128.")
	embed.add_argument("--epochs", type=int, default=10, help="Number of epochs. Default is 10.")
	embed.add_argument("--min-count", type=int, default=5, help="Minimal structural feature count. Default is 5.")
	embed.add_argument("--wl-iterations", type=int, default=2, help="Number of Weisfeiler-Lehman iterations. Default is 2.")
	embed.add_argument("--learning-rate", type=float, default=0.025, help="Initial learning rate. Default is 0.025.")
	embed.add_argument("--down-sampling", type=float, default=0.0001, help="Down sampling rate of features. Default is 0.0001.")
	embed.add_argument("--workers", type=int, default=8, help="Number of workers. Default is 8.")
	embed.set_defaults(func=generate_embeddings)

	viz = sub.add_parser("data_viz", help="Visualize embedding spaces.")
	viz.add_argument("--output", nargs="?", default="images", help="Images path.")
	viz.add_argument("--input", default="embeddings.csv", help="Specify the embeddings to use for visualization.")
	viz.set_defaults(func=data_viz)

	args = parser.parse_args(sys.argv[1:])
	args.func(args)

if __name__ == "__main__":
	main()
