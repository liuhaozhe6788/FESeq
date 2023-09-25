#! /bin/sh
home="model_zoo"

export PYTHONPATH=${PWD}
echo ${PYTHONPATH}
cd $home/BST 

echo "=== Training BST ===" && python run_expid.py --gpu 0 --expid BST_eleme

echo "=== Training BST ===" && python run_expid.py --gpu 0 --expid BST_bundle
