for CPU_CORES in 1 2 4 8 16 32
do
	oarsub -t exotic -p yeti -l core=$CPU_CORES,walltime=3 /home/tryckeboer/benchmark/activate_environment.sh
done
