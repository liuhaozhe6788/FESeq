#! /bin/sh
home="model_zoo"

export PYTHONPATH=${PWD}
echo ${PYTHONPATH}
cd $home/BST 

echo "=== Training BST_eleme ===" && python run_expid.py --gpu 0 --expid BST_eleme

echo "=== Training BST_bundle ===" && python run_expid.py --gpu 0 --expid BST_bundle

echo "=== Training BST_ml10m ===" && python run_expid.py --gpu 0 --expid BST_ml10m