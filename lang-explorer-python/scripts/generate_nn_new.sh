#!/bin/bash

declare -a css_indices=("173" "174" "177" "225")
for i in "${!css_indices[@]}"
do
    ./main.py neigh3 --indices ${css_indices[$i]} --lang css --experiment-number 7 --embedding-system doc2vecgensim --output css_doc2vecgensim_${i}_5000
done

declare -a tacosched_indices=("26" "42" "50" "52")
for i in "${!tacosched_indices[@]}"
do
    ./main.py neigh3 --indices ${tacosched_indices[$i]} --lang tacosched --experiment-number 8 --embedding-system doc2vecgensim --output tacosched_doc2vecgensim_${i}_5000
done

declare -a tacoexpr_indices=("30" "1687" "65" "3760")
for i in "${!tacoexpr_indices[@]}"
do
    ./main.py neigh3 --indices ${tacoexpr_indices[$i]} --lang tacoexpr --experiment-number 5 --embedding-system doc2vecgensim --output tacoexpr_doc2vecgensim_${i}_5000
done

declare -a karel_indices=("702" "86" "96" "1419")
for i in "${!karel_indices[@]}"
do
    ./main.py neigh3 --indices ${karel_indices[$i]} --lang karel --experiment-number 7 --embedding-system doc2vecgensim --output karel_doc2vecgensim_${i}_5000
done
