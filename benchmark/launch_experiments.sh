source ~/.bashrc
echo "START"
for ENVIRONMENT in mkl openblas blis netlib
do
                    for PENALTY in l1 l2
                    do
                        for LAMBDA in 0.1 0.01 0.001 0.0001 0.00001 0.000001 0.0000001
                        do
    conda activate cyanure_benchmark_${ENVIRONMENT}
    python benchmark_rework.py --dataset covtype --penalty ${PENALTY} --lambda_1 ${LAMBDA}
    python benchmark_rework.py --dataset alpha --penalty ${PENALTY} --lambda_1 ${LAMBDA}
    python benchmark_rework.py --dataset real-sim --penalty ${PENALTY} --lambda_1 ${LAMBDA}
    python benchmark_rework.py --dataset epsilon --penalty ${PENALTY} --lambda_1 ${LAMBDA}
    python benchmark_rework.py --dataset ocr --penalty ${PENALTY} --lambda_1 ${LAMBDA}
    python benchmark_rework.py --dataset rcv1 --penalty ${PENALTY} --lambda_1 ${LAMBDA}
    python benchmark_rework.py --dataset webspam --penalty ${PENALTY} --lambda_1 ${LAMBDA}
    python benchmark_rework.py --dataset kddb --penalty ${PENALTY} --lambda_1 ${LAMBDA}
    python benchmark_rework.py --dataset criteo --penalty ${PENALTY} --lambda_1 ${LAMBDA}
    python benchmark_rework.py --dataset ckn_mnist --penalty ${PENALTY} --lambda_1 ${LAMBDA} --loss multiclass-logistic
    python benchmark_rework.py --dataset ckn_svhn --penalty ${PENALTY} --lambda_1 ${LAMBDA} --loss multiclass-logistic
        done
    done
done
echo "END"