#! /bin/sh
home="model_zoo"

export PYTHONPATH=${PWD}
echo ${PYTHONPATH}
cd $home/DIN 

echo "=== Training DIN ===" && python run_expid.py  --gpu 0 --expid DIN_eleme

echo "=== Training DIN ===" && python run_expid.py  --gpu 0 --expid DIN_bundle