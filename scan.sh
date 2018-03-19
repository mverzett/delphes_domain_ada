#! /bin/bash

set -o nounset
set -o errexit

ind=/data/ml/mverzett/pheno_domAda/smearing_x5/
outd=../smearing_5x_75epochs/
lmbs=0.005,0.01,0.04,0.1,0.3
weights=10,30,50,100,300

python scan.py $outd $ind 0 --gpu=1 --gpufraction=0.25 --lr=0.0005,0.001 --lmb=$lmbs --weight=$weights &> 0.0.log &
python scan.py $outd $ind 1 --gpu=1 --gpufraction=0.25 --lr=0.0005,0.001 --lmb=$lmbs --weight=$weights &> 0.1.log &
python scan.py $outd $ind 2 --gpu=1 --gpufraction=0.25 --lr=0.0005,0.001 --lmb=$lmbs --weight=$weights &> 0.2.log &
python scan.py $outd $ind 3 --gpu=2 --gpufraction=0.25 --lr=0.0005,0.001 --lmb=$lmbs --weight=$weights &> 0.3.log &
python scan.py $outd $ind 4 --gpu=2 --gpufraction=0.25 --lr=0.0005,0.001 --lmb=$lmbs --weight=$weights &> 0.4.log &
python scan.py $outd $ind 5 --gpu=2 --gpufraction=0.25 --lr=0.0005,0.001 --lmb=$lmbs --weight=$weights &> 0.5.log &
python scan.py $outd $ind 6 --gpu=3 --gpufraction=0.25 --lr=0.0005,0.001 --lmb=$lmbs --weight=$weights &> 0.6.log &
python scan.py $outd $ind 7 --gpu=3 --gpufraction=0.25 --lr=0.0005,0.001 --lmb=$lmbs --weight=$weights &> 0.7.log &
python scan.py $outd $ind 8 --gpu=3 --gpufraction=0.25 --lr=0.0005,0.001 --lmb=$lmbs --weight=$weights &> 0.8.log &
python scan.py $outd $ind 9 --gpu=4 --gpufraction=0.25 --lr=0.0005,0.001 --lmb=$lmbs --weight=$weights &> 0.9.log &


python scan.py $outd $ind 0 --gpu=4 --gpufraction=0.25 --lr=0.003,0.005 --lmb=$lmbs --weight=$weights &> 1.0.log &
python scan.py $outd $ind 1 --gpu=4 --gpufraction=0.25 --lr=0.003,0.005 --lmb=$lmbs --weight=$weights &> 1.1.log &
python scan.py $outd $ind 2 --gpu=5 --gpufraction=0.25 --lr=0.003,0.005 --lmb=$lmbs --weight=$weights &> 1.2.log &
python scan.py $outd $ind 3 --gpu=5 --gpufraction=0.25 --lr=0.003,0.005 --lmb=$lmbs --weight=$weights &> 1.3.log &
python scan.py $outd $ind 4 --gpu=5 --gpufraction=0.25 --lr=0.003,0.005 --lmb=$lmbs --weight=$weights &> 1.4.log &
python scan.py $outd $ind 5 --gpu=6 --gpufraction=0.25 --lr=0.003,0.005 --lmb=$lmbs --weight=$weights &> 1.5.log &
python scan.py $outd $ind 6 --gpu=6 --gpufraction=0.25 --lr=0.003,0.005 --lmb=$lmbs --weight=$weights &> 1.6.log &
python scan.py $outd $ind 7 --gpu=6 --gpufraction=0.25 --lr=0.003,0.005 --lmb=$lmbs --weight=$weights &> 1.7.log &
python scan.py $outd $ind 8 --gpu=7 --gpufraction=0.25 --lr=0.003,0.005 --lmb=$lmbs --weight=$weights &> 1.8.log &
python scan.py $outd $ind 9 --gpu=7 --gpufraction=0.25 --lr=0.003,0.005 --lmb=$lmbs --weight=$weights &> 1.9.log &

wait