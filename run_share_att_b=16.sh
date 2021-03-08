#!/usr/bin/env bash
gpu=$1
node=$2

CUDA_VISIBLE_DEVICES=${gpu} python -m torch.distributed.launch \
	--nproc_per_node=${node} train_dense_encoder.py \
	--max_grad_norm 2.0 \
	--encoder_model_type hf_bert_share_att \
	--pretrained_model_cfg /home/share/jiaofangkai/pretrained-models/bert-base-uncased \
	--seed 12345 \
	--sequence_length 256 \
	--warmup_steps 1237 \
	--batch_size 16 \
	--do_lower_case \
	--train_file /home/share/jiaofangkai/DPR_data/data/retriever/nq-train.json \
	--dev_file /home/share/jiaofangkai/DPR_data/data/retriever/nq-dev.json \
	--output_dir experiments_share/dpr_bert_retriever_share_att.b.16 \
	--learning_rate 1e-05 \
	--num_train_epochs 30 \
	--dev_batch_size 16 \
	--fp16 \
	--fp16_opt_level O2 \
	--val_av_rank_start_epoch 100000 \
	--global_loss_buf_sz 33554432
