#! /bin/bash

set -o nounset
set -o errexit

rm -f *.*.log
nprocs=0

dirs1='/data/ml/jkiesele/pheno_domAda/numpyx2:../smearing_2x_75epochs_balanced_samples'
disr2='/data/ml/jkiesele/pheno_domAda/numpyx5:../smearing_5x_75epochs_balanced_samples'
for dirvals in $dirs1 $disr2; do
		indir="${dirvals%:*}"
		outdir="${dirvals#*:}"
		for svopts in ':'; do #'--addsv:_SV' ':'; do
				svopt="${svopts%:*}"
				postfix="${svopts#*:}"
				for method in 'MC_training' 'data_training' 'domain_adaptation_two_samples_w50_l.04' 'domain_adaptation_two_samples_lr0.0005_w300_l0.04'; do		
						for i in $(seq 10); do
								igpu=$(($nprocs%2+6))
				## if [ $igpu -eq 6 ]
				## then
				## 		igpu=7
				## fi
								mkdir -p $outdir$postfix
								python AdaptMeDelpes.py $outdir$postfix/$method/$i -i $indir $method --gpu=$igpu $svopt --gpufraction=0.20 &> $outdir$postfix/$method.$i.log &
								nprocs=$(($nprocs+1))
								if [ $nprocs -eq 8 ]
								then						
										echo waiting
										wait
										nprocs=0
								fi
						done
				done
		done
done
echo waiting
wait


for dirvals in $dirs1 $disr2; do
		indir="${dirvals%:*}"
		outdir="${dirvals#*:}"
		for svopts in ':'; do
				svopt="${svopts%:*}"
				postfix="${svopts#*:}"
				for method in 'MC_training' 'data_training' 'domain_adaptation_two_samples_w50_l.04'; do		
						python compute_averages.py $outdir$postfix/$method
				done
				python make_plots.py $outdir$postfix
		done
done
