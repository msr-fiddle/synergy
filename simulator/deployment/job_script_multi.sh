

python -m torch.distributed.launch --nproc_per_node=$num_gpu job_script_multi.py -a $arch -b $batch --workers $workers --epochs 3
