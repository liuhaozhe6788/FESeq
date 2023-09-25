#! /bin/sh
home="model_zoo"

export PYTHONPATH=${PWD}
echo ${PYTHONPATH}

cd $home/SASRec

echo "=== Training SASRec ===" && python run_expid.py --gpu 0 --expid SASRec_eleme

echo "=== Training SASRec ===" && python run_expid.py --gpu 0 --expid SASRec_bundle
