#!/usr/bin/env bash

python -m torch.distributed.launch \
	--nproc_per_node=2 train_dense_encoder.py \
	--max_grad_norm 2.0 \
	--encoder_model_type hf_bert_share_att \
	--pretrained_model_cfg /home/share/jiaofangkai/pretrained-models/bert-base-uncased \
	--seed 12345 \
	--sequence_length 256 \
	--warmup_steps 1237 \
	--batch_size 8 \
	--do_lower_case \
	--train_file /home/share/jiaofangkai/DPR_data/data/retriever/nq-train.json \
	--dev_file /home/share/jiaofangkai/DPR_data/data/retriever/nq-dev.json \
	--output_dir experiments_share/dpr_bert_retriever_share_att_test \
	--learning_rate 1e-05 \
	--num_train_epochs 60 \
	--dev_batch_size 8 \
	--fp16 \
	--val_av_rank_start_epoch 100000 \
	--global_loss_buf_sz 16777216
