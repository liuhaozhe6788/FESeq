#! /bin/sh
home="model_zoo"

export PYTHONPATH=${PWD}
echo ${PYTHONPATH}

cd $home/DeepFM/DeepFM_torch 

echo "=== Training DeepFM_torch ===" && python run_expid.py --gpu 0 --expid DeepFM_eleme

echo "=== Training DeepFM_torch ===" && python run_expid.py --gpu 0 --expid DeepFM_bundle
