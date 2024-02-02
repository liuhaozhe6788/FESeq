#! /bin/sh
home="model_zoo"

export PYTHONPATH=${PWD}
echo ${PYTHONPATH}
cd $home/ONN/ONN_torch 

echo "=== Training ONNv2_eleme ===" && python run_expid.py --gpu 0 --expid ONNv2_eleme

echo "=== Training ONNv2_bundle ===" && python run_expid.py --gpu 0 --expid ONNv2_bundle

echo "=== Training ONNv2_ml10m ===" && python run_expid.py --gpu 0 --expid ONNv2_ml10m
