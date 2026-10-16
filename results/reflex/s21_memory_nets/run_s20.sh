cd work/p4_s21/src; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 S20_ARMS=net_gru,net_peep,net_mog,net_tfm,net_memnet
for lr in 3e-3 1e-2; do S20_LSTM=$lr,600,16 S20_TAG=_s21_lr$lr python s20_timecourse.py ../s20_tune 101,102,103,104,105 10 4; done > ../s20_tune.log 2>&1
for lr in 1e-2 3e-3; do for n in 10 3; do S20_LSTM=$lr,600,16 S20_TAG=_s21_lr$lr python s20_timecourse.py ../s20 $(seq -s, 1 20) $n 4; done; done > ../s20.log 2>&1
date -u > ../s20_done.txt
