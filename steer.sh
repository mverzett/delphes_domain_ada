#! /bin/bash

rm *.*.log
nprocs=0
for method in 'MC_training' 'data_training' 'domain_adaptation_two_samples_w50_l.04'; do		
		for i in $(seq 10); do
				igpu=$(($nprocs%5+2))
				if [ $igpu -eq 6 ]
				then
						igpu=7
				fi
				python AdaptMeDelpes.py ../domada_50_epochs_newsample_sv/$method/$i $method --gpu=$igpu --gpufraction=0.25 &> $method.$i.log &
				nprocs=$(($nprocs+1))
				if [ $nprocs -eq 15 ]
				then						
						echo waiting
						wait
						nprocs=0
				fi
		done
done
echo waiting
wait

for method in 'MC_training' 'data_training' 'domain_adaptation_two_samples_w50_l.04'; do		
		python compute_averages.py ../domada_50_epochs_newsample_sv/$method
done
