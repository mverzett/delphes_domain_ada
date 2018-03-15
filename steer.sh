#! /bin/bash

rm *.*.log
nprocs=0
for method in 'domain_adaptation_two_samples' 'MC_training' 'data_training' 'domain_adaptation_two_samples_w50_l.25' 'domain_adaptation_two_samples_w50_l.04' 'domain_adaptation_two_samples_w25_l.5' 'domain_adaptation_two_samples_w05_l1'; do		
		for i in $(seq 10); do
				igpu=$(($nprocs%5+2))
				python AdaptMeDelpes.py ../domada_50_epochs_newsample/$method/$i $method --gpu=$igpu --gpufraction=0.25 &> $method.$i.log &
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

for method in 'domain_adaptation_two_samples' 'MC_training' 'data_training' 'domain_adaptation_two_samples_w50_l.25' 'domain_adaptation_two_samples_w50_l.04' 'domain_adaptation_two_samples_w25_l.5' 'domain_adaptation_two_samples_w05_l1'; do		
		python compute_averages.py ../domada_50_epochs_newsample/$method
done
