#!/bin/bash

declare -a css_indices=("173" "174" "177" "225")
for i in "${!css_indices[@]}"
do
    ./main.py neigh2 --indices ${css_indices[$i]} --input results/embeddings_128_15000_css_gensim.csv --output css${i}-15000
done

declare -a tacosched_indices=("26" "42" "50" "52")
for i in "${!tacosched_indices[@]}"
do
    ./main.py neigh2 --indices ${tacosched_indices[$i]} --input results/embeddings_128_15000_tacosched_gensim.csv --output tacosched${i}-15000
done

declare -a tacoexpr_indices=("30" "1687" "65" "3760")
for i in "${!tacoexpr_indices[@]}"
do
    ./main.py neigh2 --indices ${tacoexpr_indices[$i]} --input results/embeddings_128_15000_tacoexpr_gensim.csv --output tacoexpr${i}-15000
done

declare -a karel_indices=("702" "86" "96" "1419")
for i in "${!karel_indices[@]}"
do
    ./main.py neigh2 --indices ${karel_indices[$i]} --input results/embeddings_128_5000_karel_gensim.csv --output karel${i}-5000
done
