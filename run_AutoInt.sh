#! /bin/sh
home="model_zoo"

export PYTHONPATH=${PWD}
echo ${PYTHONPATH}
cd $home/AutoInt 

echo "=== Training AutoInt_eleme ===" && python run_expid.py --gpu 0 --expid AutoInt_eleme

echo "=== Training AutoInt_bundle ===" && python run_expid.py --gpu 0 --expid AutoInt_bundle

echo "=== Training AutoInt_ml10m ===" && python run_expid.py --gpu 0 --expid AutoInt_ml10m 