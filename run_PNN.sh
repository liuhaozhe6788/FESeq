#! /bin/sh
home="model_zoo"

export PYTHONPATH=${PWD}
echo ${PYTHONPATH}

cd $home/PNN

echo "=== Training PNN ===" && python run_expid.py --gpu 0 --expid PNN_eleme

echo "=== Training PNN ===" && python run_expid.py --gpu 0 --expid PNN_bundle
