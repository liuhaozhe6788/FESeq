#! /bin/sh
home="model_zoo"

export PYTHONPATH=${PWD}
echo ${PYTHONPATH}
cd $home/DMR

echo "=== Training DMR_eleme ===" && python run_expid.py  --gpu 0  --expid DMR_eleme
echo "=== Training DMR_bundle ===" && python run_expid.py  --gpu 0  --expid DMR_bundle
echo "=== Training DMR_ml10m ===" && python run_expid.py  --gpu 0  --expid DMR_ml10m