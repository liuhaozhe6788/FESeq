#! /bin/sh
home="model_zoo"

export PYTHONPATH=${PWD}
echo ${PYTHONPATH}
cd $home/FinalMLP 

echo "=== Training FinalMLP ===" && python run_expid.py --gpu 0 --expid DualMLP_eleme

echo "=== Training FinalMLP ===" && python run_expid.py --gpu 0 --expid FinalMLP_eleme

echo "=== Training FinalMLP ===" && python run_expid.py --gpu 0 --expid DualMLP_bundle

echo "=== Training FinalMLP ===" && python run_expid.py --gpu 0 --expid FinalMLP_bundle