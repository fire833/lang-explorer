#!/usr/bin/env python

from src.utils.client import generate, GenerateParams, GenerateResults, CSSLanguageParameters, TacoScheduleParameters, TacoExpressionParameters
from argparse import ArgumentParser	
import sys

def main():
	parser = ArgumentParser(description="Lang-Explorer Python stuff")

	args = parser.parse_args(sys.argv[1:])
	# args.func(args)

	generate("http://localhost:8080", "css", "mc", 
		GenerateParams(100, True, True, False, 3, 
		css=CSSLanguageParameters(["foo", "bar", "baz"], ["1", "2", "3"]),
		# taco_expression=TacoExpressionParameters(),
		# taco_schedule=TacoScheduleParameters(),
		# TacoScheduleParameters(index_variables=["i"], workspace_index_variables=["j"], fused_index_variables=["k"], split_factor_variables=["l"], divide_factor_variables=["m"], unroll_factor_variables=["n"]),
		# TacoExpressionParameters([], []),
		))

if __name__ == "__main__":
	main()
