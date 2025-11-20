#!/bin/bash

montage images/karel/doc2vecgensim_histogram.png images/karel/AST_histogram.png images/karel/snowflake-arctic-embed_137m_histogram.png images/karel/nomic-embed-text_histogram.png -tile 2x2 -geometry 600x400+0+0 images/histograms_karel_presentation.jpeg

montage images/tacosched/nopartials/doc2vecgensim_histogram.png images/tacosched/nopartials/AST_histogram.png images/tacosched/nopartials/snowflake-arctic-embed_137m_histogram.png images/tacosched/nopartials/mxbai-embed-large_histogram.png -tile 2x2 -geometry 600x400+0+0 images/histograms_tacosched_presentation.jpeg
