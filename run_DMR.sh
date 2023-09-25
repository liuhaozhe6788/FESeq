#! /bin/sh
home="model_zoo"

export PYTHONPATH=${PWD}
echo ${PYTHONPATH}
cd $home/DMR

echo "=== Training DMR ===" && python run_expid.py  --gpu 0  --expid DMR_eleme
echo "=== Training DMR ===" && python run_expid.py  --gpu 0  --expid DMR_bundle
