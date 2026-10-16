cd work/p4_s21; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
date -u > main_start.txt
xargs -P 8 -I{} sh -c '{}' < main.txt > main.log 2>&1; date -u > main_done.txt
