#!/bin/bash

./main.py dataviz --input results/embeddings_128_10000_tacosched_gensim.csv
./main.py dataviz --input results/embeddings_128_10000_tacosched_nopartials_gensim.csv

./main.py dataviz --input results/embeddings_128_10000_tacoexpr_gensim.csv
./main.py dataviz --input results/embeddings_128_10000_tacoexpr_nopartials_gensim.csv

./main.py dataviz --input results/embeddings_128_5000_karel_gensim.csv
./main.py dataviz --input results/embeddings_128_5000_karel_nopartials_gensim.csv

./main.py dataviz --input results/embeddings_128_10000_css_gensim.csv
./main.py dataviz --input results/embeddings_128_10000_css_nopartials_gensim.csv
