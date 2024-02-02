#! /bin/sh
home="model_zoo"

export PYTHONPATH=${PWD}
echo ${PYTHONPATH}

cd $home/JointCTR

echo "=== Training JointCTR_eleme ===" && python run_expid.py --gpu 0 --expid JointCTR_eleme

echo "=== Training JointCTR_bundle ===" && python run_expid.py --gpu 0 --expid JointCTR_bundle

echo "=== Training JointCTR_ml10m ===" && python run_expid.py --gpu 0 --expid JointCTR_ml10m