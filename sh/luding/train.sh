#!/bin/bash

# 启动 predict_server.py 并显示日志
python server.py \
	--config configs/mmld/luding/server_MMLD.yml \
	--save_dir output/luding_new/server \
	--do_eval \
	--log_iters 50 \
	--save_interval 200 &
sleep 10  # 等待 10 秒

# 启动 predict_client1.py，不输出日志
python client1.py \
	--config configs/mmld/luding/client1_MMLD.yml \
	--save_dir output/luding_new/client1 \
	--do_eval \
	--log_iters 50 \
	--save_interval 200 > /dev/null 2>&1 &
sleep 10  # 等待 10 秒

# 启动 predict_client2.py，不输出日志
python client2.py \
	--config configs/mmld/luding/client2_MMLD.yml \
	--save_dir output/luding_new/client2 \
	--do_eval \
	--log_iters 50 \
	--save_interval 200 > /dev/null 2>&1 &
sleep 10  # 等待 10 秒

# 启动 predict_client3.py，不输出日志
python client3.py \
	--config configs/mmld/luding/client3_MMLD.yml \
	--save_dir output/luding_new/client3 \
	--do_eval \
	--log_iters 50 \
	--save_interval 200 > /dev/null 2>&1 &

# 等待所有后台任务完成
wait
