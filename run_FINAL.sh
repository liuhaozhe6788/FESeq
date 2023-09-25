#! /bin/sh
home="model_zoo"

export PYTHONPATH=${PWD}
echo ${PYTHONPATH}
cd $home/FINAL 

echo "=== Training FINAL ===" && python run_expid.py --gpu 0 --expid FINAL_eleme

echo "=== Training FINAL ===" && python run_expid.py --gpu 0 --expid FINAL_bundle
