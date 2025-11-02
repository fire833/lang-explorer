#!/bin/bash

cargo build --release --bin explorer

./target/release/explorer generate -l tacosched -e wmc -c configs/similarity_nopartials2.json
./target/release/explorer generate -l tacoexpr -e wmc -c configs/similarity_nopartials2.json
./target/release/explorer generate -l karel -e wmc -c configs/similarity_nopartials2.json
./target/release/explorer generate -l css -e wmc -c configs/similarity_nopartials2.json
