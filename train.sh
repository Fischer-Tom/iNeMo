#!/bin/bash
source /home/fischer/.bashrc
ROOT="/home/fischer/iNeMo/"

cd $ROOT

python "src/train.py" "$@"
