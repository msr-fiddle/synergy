#rm gpu_utillization.csv
nvidia-smi --query-gpu=power.draw,utilization.gpu -l 10 --format=csv >> gpu_util.csv 
#while sleep 1; 
#do (nvidia-smi --query-gpu=power.draw,utilization.gpu --format=csv >> gpu_utillization.csv) 
#done
