#! /bin/sh
home="model_zoo"

export PYTHONPATH=${PWD}
echo ${PYTHONPATH}
cd $home/FINAL 

echo "=== Training FINAL_eleme ===" && python run_expid.py --gpu 0 --expid FINAL_eleme

echo "=== Training FINAL_bundle ===" && python run_expid.py --gpu 0 --expid FINAL_bundle

echo "=== Training FINAL_ml10m ===" && python run_expid.py --gpu 0 --expid FINAL_ml10m