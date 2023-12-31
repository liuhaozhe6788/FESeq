#! /bin/sh
home="model_zoo"

export PYTHONPATH=${PWD}
echo ${PYTHONPATH}

cd $home/FESeq

echo "=== Training FESeq===" && python run_expid.py --gpu 0 --expid FESeq_2_eleme
echo "=== Training FESeq===" && python run_expid.py --gpu 0 --expid FESeq_3_eleme
echo "=== Training FESeq===" && python run_expid.py --gpu 0 --expid FESeq_4_eleme
echo "=== Training FESeq===" && python run_expid.py --gpu 0 --expid FESeq_5_eleme
echo "=== Training FESeq===" && python run_expid.py --gpu 0 --expid FESeq_6_eleme
echo "=== Training FESeq===" && python run_expid.py --gpu 0 --expid FESeq_7_eleme
echo "=== Training FESeq===" && python run_expid.py --gpu 0 --expid FESeq_8_eleme
echo "=== Training FESeq===" && python run_expid.py --gpu 0 --expid FESeq_9_eleme
echo "=== Training FESeq===" && python run_expid.py --gpu 0 --expid FESeq_10_eleme
echo "=== Training FESeq===" && python run_expid.py --gpu 0 --expid FESeq_11_eleme
echo "=== Training FESeq===" && python run_expid.py --gpu 0 --expid FESeq_12_eleme

echo "=== Training FESeq===" && python run_expid.py --gpu 0 --expid FESeq_2_bundle
echo "=== Training FESeq===" && python run_expid.py --gpu 0 --expid FESeq_3_bundle
echo "=== Training FESeq===" && python run_expid.py --gpu 0 --expid FESeq_4_bundle
echo "=== Training FESeq===" && python run_expid.py --gpu 0 --expid FESeq_5_bundle
echo "=== Training FESeq===" && python run_expid.py --gpu 0 --expid FESeq_6_bundle
echo "=== Training FESeq===" && python run_expid.py --gpu 0 --expid FESeq_7_bundle
echo "=== Training FESeq===" && python run_expid.py --gpu 0 --expid FESeq_8_bundle
echo "=== Training FESeq===" && python run_expid.py --gpu 0 --expid FESeq_9_bundle
echo "=== Training FESeq===" && python run_expid.py --gpu 0 --expid FESeq_10_bundle
echo "=== Training FESeq===" && python run_expid.py --gpu 0 --expid FESeq_11_bundle
echo "=== Training FESeq===" && python run_expid.py --gpu 0 --expid FESeq_12_bundle