#!/bin/bash

declare -a css_indices=("173" "174" "177" "225")
for i in "${!css_indices[@]}"
do
    ./main.py neigh3 --indices ${css_indices[$i]} --language css --experiment-number --embedding-engine --output css${i}_15000
done

declare -a tacosched_indices=("26" "42" "50" "52")
for i in "${!tacosched_indices[@]}"
do
    ./main.py neigh3 --indices ${tacosched_indices[$i]} --input results/embeddings_128_15000_tacosched_gensim.csv --output tacosched${i}_15000
done

declare -a tacoexpr_indices=("30" "1687" "65" "3760")
for i in "${!tacoexpr_indices[@]}"
do
    ./main.py neigh3 --indices ${tacoexpr_indices[$i]} --input results/embeddings_128_15000_tacoexpr_gensim.csv --output tacoexpr${i}_15000
done

declare -a karel_indices=("702" "86" "96" "1419")
for i in "${!karel_indices[@]}"
do
    ./main.py neigh3 --indices ${karel_indices[$i]} --input results/embeddings_128_5000_karel_gensim.csv --output karel${i}_5000
done
