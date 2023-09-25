#! /bin/sh
home="model_zoo"

export PYTHONPATH=${PWD}
echo ${PYTHONPATH}

cd $home/WideDeep

echo "=== Training WideDeep_torch ===" && python run_expid.py --gpu 0 --expid WideDeep_eleme

echo "=== Training WideDeep_torch ===" && python run_expid.py --gpu 0 --expid WideDeep_bundle

