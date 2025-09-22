#!/bin/bash

./main.py embedgen --count 10000 --language tacosched --seed 10 --partials false
mv results/embeddings_128_10000_tacosched_gensim.csv results/embeddings_128_10000_tacosched_nopartials_gensim.csv
./main.py embedgen --count 10000 --language tacoexpr --seed 10 --partials false
mv results/embeddings_128_10000_tacoexpr_gensim.csv results/embeddings_128_10000_tacoexpr_nopartials_gensim.csv
./main.py embedgen --count 5000 --language karel --seed 10 --partials false
mv results/embeddings_128_5000_karel_gensim.csv results/embeddings_128_5000_karel_nopartials_gensim.csv
./main.py embedgen --count 10000 --language css --seed 10 --partials false
mv results/embeddings_128_10000_css_gensim.csv results/embeddings_128_10000_css_nopartials_gensim.csv

./main.py embedgen --count 10000 --language tacosched --seed 10
./main.py embedgen --count 10000 --language tacoexpr --seed 10
./main.py embedgen --count 5000 --language karel --seed 10
./main.py embedgen --count 10000 --language css --seed 10
