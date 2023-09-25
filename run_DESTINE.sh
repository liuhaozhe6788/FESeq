#! /bin/sh
home="model_zoo"

export PYTHONPATH=${PWD}
echo ${PYTHONPATH}
cd $home/DESTINE 

echo "=== Training DESTINE ===" && python run_expid.py  --gpu 0 --expid DESTINE_eleme

echo "=== Training DESTINE ===" && python run_expid.py  --gpu 0 --expid DESTINE_bundle
