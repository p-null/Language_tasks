#!/usr/bin/env bash
set -e

wget -P data/raw -O data/raw/multinli_1.0.zip https://www.nyu.edu/projects/bowman/multinli/multinli_1.0.zip
unzip data/raw/multinli_1.0.zip -d data/raw/

parent_path = $( cd "$(dirname "${BASH_SOURCE}")" ; pwd -P )
cd "$parent_path"
mkdir -p ../../../data/processed/multinli/
