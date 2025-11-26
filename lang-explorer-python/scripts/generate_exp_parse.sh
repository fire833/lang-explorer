#!/bin/bash

./main.py expparse --input results/tacosched/6/experiments.json --output images/tacosched
./main.py expparse --input results/tacoexpr/3/experiments.json --output images/tacoexpr
./main.py expparse --input results/karel/6/experiments.json --output images/karel
./main.py expparse --input results/css/5/experiments.json --output images/css

./main.py expparse --input results/tacosched/7/experiments.json --output images/tacosched/nopartials
./main.py expparse --input results/tacoexpr/4/experiments.json --output images/tacoexpr/nopartials
./main.py expparse --input results/karel/5/experiments.json --output images/karel/nopartials
./main.py expparse --input results/css/6/experiments.json --output images/css/nopartials

prefix="images/karel/"
montage ${prefix}doc2vecgensim_histogram.svg ${prefix}nomic-embed-text_histogram.svg ${prefix}mxbai-embed-large_histogram.svg ${prefix}snowflake-arctic-embed_137m_histogram.svg ${prefix}nopartials/doc2vecgensim_histogram.svg ${prefix}nopartials/nomic-embed-text_histogram.svg ${prefix}nopartials/mxbai-embed-large_histogram.svg ${prefix}nopartials/snowflake-arctic-embed_137m_histogram.svg -tile 2x4 -geometry 600x400+0+0 ${prefix}histograms_karel.svg

montage ${prefix}AST_histogram.svg ${prefix}nopartials/AST_histogram.svg -tile 2x1 -geometry 600x400+0+0 ${prefix}AST_histograms_karel.svg

prefix="images/css/"
montage ${prefix}doc2vecgensim_histogram.svg ${prefix}nomic-embed-text_histogram.svg ${prefix}mxbai-embed-large_histogram.svg ${prefix}snowflake-arctic-embed_137m_histogram.svg ${prefix}nopartials/doc2vecgensim_histogram.svg ${prefix}nopartials/nomic-embed-text_histogram.svg ${prefix}nopartials/mxbai-embed-large_histogram.svg ${prefix}nopartials/snowflake-arctic-embed_137m_histogram.svg -tile 2x4 -geometry 600x400+0+0 ${prefix}histograms_css.svg

montage ${prefix}AST_histogram.svg ${prefix}nopartials/AST_histogram.svg -tile 2x1 -geometry 600x400+0+0 ${prefix}AST_histograms_css.svg

prefix="images/tacosched/"
montage ${prefix}doc2vecgensim_histogram.svg ${prefix}nomic-embed-text_histogram.svg ${prefix}mxbai-embed-large_histogram.svg ${prefix}snowflake-arctic-embed_137m_histogram.svg ${prefix}nopartials/doc2vecgensim_histogram.svg ${prefix}nopartials/nomic-embed-text_histogram.svg ${prefix}nopartials/mxbai-embed-large_histogram.svg ${prefix}nopartials/snowflake-arctic-embed_137m_histogram.svg -tile 2x4 -geometry 600x400+0+0 ${prefix}histograms_tacosched.jpeg

montage ${prefix}AST_histogram.svg ${prefix}nopartials/AST_histogram.svg -tile 2x1 -geometry 600x400+0+0 ${prefix}AST_histograms_tacosched.svg

prefix="images/tacoexpr/"
montage ${prefix}doc2vecgensim_histogram.svg ${prefix}nomic-embed-text_histogram.svg ${prefix}mxbai-embed-large_histogram.svg ${prefix}snowflake-arctic-embed_137m_histogram.svg ${prefix}nopartials/doc2vecgensim_histogram.svg ${prefix}nopartials/nomic-embed-text_histogram.svg ${prefix}nopartials/mxbai-embed-large_histogram.svg ${prefix}nopartials/snowflake-arctic-embed_137m_histogram.svg -tile 2x4 -geometry 600x400+0+0 ${prefix}histograms_tacoexpr.jpeg

montage ${prefix}AST_histogram.svg ${prefix}nopartials/AST_histogram.svg -tile 2x1 -geometry 600x400+0+0 ${prefix}AST_histograms_tacoexpr.svg
