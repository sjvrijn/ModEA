#!/bin/bash
num_threads=60


for dim in 5 #20 #15 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100
do

    for fid in 22 23 24 # 7
    do

        for part in 0 1 2 3
        do
        
            mpirun -np $num_threads python2 ~/src/modular-cma-es-framework/main.py $dim $fid 1 $part > ~/BF_out_$(($dim))dim_f$(($fid))_$(($part)).txt 2>&1
        
        done

    done

done
