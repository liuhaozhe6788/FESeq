#! /bin/sh
home="model_zoo"

export PYTHONPATH=${PWD}
echo ${PYTHONPATH}
cd $home/DESTINE 

echo "=== Training DESTINE_eleme ===" && python run_expid.py  --gpu 0 --expid DESTINE_eleme

echo "=== Training DESTINE_bundle ===" && python run_expid.py  --gpu 0 --expid DESTINE_bundle

echo "=== Training DESTINE_ml10m ===" && python run_expid.py --gpu 0 --expid DESTINE_ml10m 