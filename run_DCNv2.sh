#! /bin/sh
home="model_zoo"

export PYTHONPATH=${PWD}
echo ${PYTHONPATH}
cd $home/DCNv2 

echo "=== Training DCNv2 ===" && python run_expid.py --gpu 0 --expid DCNv2_eleme

echo "=== Training DCNv2 ===" && python run_expid.py --gpu 0 --expid DCNv2_bundle

