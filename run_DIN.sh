#! /bin/sh
home="model_zoo"

export PYTHONPATH=${PWD}
echo ${PYTHONPATH}
cd $home/DIN 

echo "=== Training DIN_eleme ===" && python run_expid.py  --gpu 0 --expid DIN_eleme

echo "=== Training DIN_bundle ===" && python run_expid.py  --gpu 0 --expid DIN_bundle

echo "=== Training DIN_ml10m ===" && python run_expid.py  --gpu 0 --expid DIN_ml10m