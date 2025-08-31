#!/bin/bash

./main.py neigh --count 3 --indices 84,199 --input results/embeddings_128_10_15000_css.csv --output css_15000
./main.py neigh --count 3 --indices 213,225 --input results/embeddings_128_10_15000_css.csv --output css2_15000
./main.py neigh --count 3 --indices 108,2268 --input results/embeddings_128_10_15000_tacosched.csv --output tacosched_15000
./main.py neigh --count 3 --indices 179,326 --input results/embeddings_128_10_15000_tacosched.csv --output tacosched2_15000
./main.py neigh --count 3 --indices 30,1687 --input results/embeddings_128_10_15000_tacoexpr.csv --output tacoexpr_15000
./main.py neigh --count 3 --indices 3678,3760 --input results/embeddings_128_10_15000_tacoexpr.csv --output tacoexpr2_15000
./main.py neigh --count 3 --indices 702,703 --input results/embeddings_128_10_5000_karel.csv --output karel_5000
./main.py neigh --count 3 --indices 1418,1419 --input results/embeddings_128_10_5000_karel.csv --output karel2_5000
