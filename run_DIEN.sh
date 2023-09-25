#! /bin/sh
home="model_zoo"

export PYTHONPATH=${PWD}
echo ${PYTHONPATH}
cd $home/DIEN

echo "=== Training DIEN ===" &&  python run_expid.py  --gpu 0 --expid DIEN_eleme

echo "=== Training DIEN ===" &&  python run_expid.py  --gpu 0 --expid DIEN_bundle

