source ~/.bashrc
for ENVIRONMENT in mkl openblas blis netlib
do
    conda activate cyanure_benchmark_${ENVIRONMENT}
    python benchmark_rework.py
done
