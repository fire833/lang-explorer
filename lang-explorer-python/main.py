#!/usr/bin/env python

from src.utils.openapi_client.api.default_api import DefaultApi
from argparse import ArgumentParser	
import sys

def main():
	parser = ArgumentParser(description="Lang-Explorer Python stuff")

	args = parser.parse_args(sys.argv[1:])
	args.func(args)

if __name__ == "__main__":
	main()
