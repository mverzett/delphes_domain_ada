#! /bin/bash

nprocs=0
for method in 'domain_adaptation_two_samples' 'MC_training' 'data_training'; do		
		for i in $(seq 10); do
				igpu=$(($nprocs%4+2))
				python AdaptMeDelpes.py ../$method/$i $method --gpu=$igpu --gpufraction=0.20 &> $method.$i.log &
				nprocs=$(($nprocs+1))
				if [ $nprocs -eq 12 ]
				then						
						echo waiting
						wait
						nprocs=0
				fi
		done
done
echo waiting
wait

for method in 'domain_adaptation_two_samples' 'MC_training' 'data_training'; do
		python compute_averages.py ../$method
done
