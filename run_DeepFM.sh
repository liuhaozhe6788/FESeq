#! /bin/sh
home="model_zoo"

export PYTHONPATH=${PWD}
echo ${PYTHONPATH}

cd $home/DeepFM/DeepFM_torch 

echo "=== Training DeepFM_eleme ===" && python run_expid.py --gpu 0 --expid DeepFM_eleme

echo "=== Training DeepFM_bundle ===" && python run_expid.py --gpu 0 --expid DeepFM_bundle

echo "=== Training DeepFM_ml10m ===" && python run_expid.py --gpu 0 --expid DeepFM_ml10m 