#! /bin/sh
home="model_zoo"

export PYTHONPATH=${PWD}
echo ${PYTHONPATH}
cd $home/ONN/ONN_torch 

echo "=== Training ONNv2 ===" && python run_expid.py --gpu 0 --expid ONNv2_eleme

echo "=== Training ONNv2 ===" && python run_expid.py --gpu 0 --expid ONNv2_bundle