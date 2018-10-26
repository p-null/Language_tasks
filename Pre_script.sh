#!/bin/bash
set -e
bash scripts/data/make_auxiliary_dirs.sh
wget -P data/external/ -O data/external/glove.840B.300d.zip https://nlp.stanford.edu/data/glove.840B.300d.zip
unzip data/external/glove.840B.300d.zip -d data/external/
#bash ./scripts/data/multinli/download_and_split_multinli.sh
bash ./scripts/data/quora/clean_and_split_quora_dataset.sh