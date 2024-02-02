#! /bin/sh
home="model_zoo"

export PYTHONPATH=${PWD}
echo ${PYTHONPATH}

cd $home/SASRec

echo "=== Training SASRec_eleme ===" && python run_expid.py --gpu 0 --expid SASRec_eleme

echo "=== Training SASRec_bundle ===" && python run_expid.py --gpu 0 --expid SASRec_bundle

echo "=== Training SASRec_ml10m ===" && python run_expid.py --gpu 0 --expid SASRec_ml10m