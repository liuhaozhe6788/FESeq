#! /bin/sh
home="model_zoo"

export PYTHONPATH=${PWD}
echo ${PYTHONPATH}
cd $home/DIEN

echo "=== Training DIEN_eleme ===" && python run_expid.py  --gpu 0 --expid DIEN_eleme

echo "=== Training DIEN_bundle ===" && python run_expid.py  --gpu 0 --expid DIEN_bundle

echo "=== Training DIEN_ml10m ===" && python run_expid.py  --gpu 0 --expid DIEN_ml10m
