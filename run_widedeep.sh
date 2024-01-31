#! /bin/sh
home="model_zoo"

export PYTHONPATH=${PWD}
echo ${PYTHONPATH}

cd $home/WideDeep

# echo "=== Training WideDeep_eleme ===" && python run_expid.py --gpu 0 --expid WideDeep_eleme

# echo "=== Training WideDeep_bundle ===" && python run_expid.py --gpu 0 --expid WideDeep_bundle

echo "=== Training WideDeep_ml10m ===" && python run_expid.py --gpu 0 --expid WideDeep_ml10m

