Usage


cd store-aware-packing/src
python profiler/offline_profiler.py --job-name job-res18 --cpu 24 --docker-img=nvidia/dali:py36_cu10.run ../models/image_classification/pytorch-imagenet.py --dali --amp --dali_cpu --max-iterations 50 --workers 3 -b 512 --data '/datadrive/mnt4/jaya/datasets/imagenet/' | tee res18out.log
