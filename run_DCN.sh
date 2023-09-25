#! /bin/sh
home="model_zoo"

export PYTHONPATH=${PWD}
echo ${PYTHONPATH}

cd $home/DCN/DCN_torch 

echo "=== Training DCN_torch ===" && python run_expid.py --gpu 0 --expid DCN_eleme

echo "=== Training DCN_torch ===" && python run_expid.py --gpu 0 --expid DCN_bundle
