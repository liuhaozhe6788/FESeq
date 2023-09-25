#! /bin/sh
home="model_zoo"

export PYTHONPATH=${PWD}
echo ${PYTHONPATH}
cd $home/AutoInt 

echo "=== Training AutoInt ===" && python run_expid.py --gpu 0 --expid AutoInt_eleme

echo "=== Training AutoInt ===" && python run_expid.py --gpu 0 --expid AutoInt_bundle
