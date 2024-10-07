#!/bin/bash

# 启动 predict_server.py 并显示日志
python predict_server.py \
	--config configs/mmld/jiuzhaigou/server_MMLD.yml \
	--save_dir pred/jiuzhaigou_new \
	--model_path output/jiuzhaigou/server/best_model/model.pdparams \
	--image_path dataset/dataset_jiuzhaigou/test_new.txt &
sleep 10  # 等待 10 秒

# 启动 predict_client1.py，不输出日志
python predict_client1.py \
	--config configs/mmld/jiuzhaigou/client1_MMLD.yml \
	--model_path output/jiuzhaigou/client1/segformer_opt.pdparams \
	--image_path dataset/dataset_jiuzhaigou/test_new.txt > /dev/null 2>&1 &
sleep 10  # 等待 10 秒

# 启动 predict_client2.py，不输出日志
python predict_client2.py \
	--config configs/mmld/jiuzhaigou/client2_MMLD.yml \
	--model_path output/jiuzhaigou/client2/hrnet_dem.pdparams \
	--image_path dataset/dataset_jiuzhaigou/test_new.txt > /dev/null 2>&1 &
sleep 10  # 等待 10 秒

# 启动 predict_client3.py，不输出日志
python predict_client3.py \
	--config configs/mmld/jiuzhaigou/client3_MMLD.yml \
	--model_path output/jiuzhaigou/client3/hrformer_base_hillshade.pdparams \
	--image_path dataset/dataset_jiuzhaigou/test_new.txt > /dev/null 2>&1 &

# 等待所有后台任务完成
wait
