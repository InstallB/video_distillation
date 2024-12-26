GPU=$1
DATA=$2
# data_path=$3
# IPC=$4

cd ..;

CUDA_VISIBLE_DEVICES=${GPU} python distill_rded.py \
--method 4m_2x2 \
--dataset ${DATA} \
--ipc 4 \
--num_eval 5 \
--epoch_eval_train 500 \
--init real \
--data_path video_distillation/UCF101/4m_2x2 \
--lr_net 0.01 \
--Iteration 5000 \
--model ConvNet3D \
--eval_mode SS \
--eval_it 500 \
--batch_real 64 \
--num_workers 4 \
# --preload
