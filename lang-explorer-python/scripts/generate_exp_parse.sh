#!/bin/bash

./main.py expparse --input results/tacosched/4/experiments.json --output images/tacosched
./main.py expparse --input results/tacoexpr/1/experiments.json --output images/tacoexpr
./main.py expparse --input results/karel/2/experiments.json --output images/karel
./main.py expparse --input results/css/3/experiments.json --output images/css

./main.py expparse --input results/tacosched/5/experiments.json --output images/tacosched/nopartials
./main.py expparse --input results/tacoexpr/2/experiments.json --output images/tacoexpr/nopartials
./main.py expparse --input results/karel/3/experiments.json --output images/karel/nopartials
./main.py expparse --input results/css/4/experiments.json --output images/css/nopartials
