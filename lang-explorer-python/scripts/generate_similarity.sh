#!/bin/bash

./main.py similarity_correlation --lang css --experiment-number 8
./main.py similarity_correlation --lang css --experiment-number 9

./main.py similarity_correlation --lang karel --experiment-number 8
./main.py similarity_correlation --lang karel --experiment-number 9

./main.py similarity_correlation --lang tacosched --experiment-number 9
./main.py similarity_correlation --lang tacosched --experiment-number 10

./main.py similarity_correlation --lang tacoexpr --experiment-number 6
./main.py similarity_correlation --lang tacoexpr --experiment-number 7
