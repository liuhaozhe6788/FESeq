#! /bin/sh
home="model_zoo"

export PYTHONPATH=${PWD}
echo ${PYTHONPATH}

cd $home/JointCTR

echo "=== Training JointCTR ===" && python run_expid.py --gpu 0 --expid JointCTR_eleme

echo "=== Training JointCTR ===" && python run_expid.py --gpu 0 --expid JointCTR_bundle