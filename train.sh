python train.py \
--experiment_name 'train_1' \
--data_root '../../Data/xxx/' \
--model_type 'model_cnn' \
--net_G 'DuRDN' \
--norm 'BN' \
--DuRDN_filters 32 \
--lr_G1 1e-5 \
--lr_G2 1e-4 \
--step_size 600 \
--gamma 1 \
--n_epochs 1000 \
--batch_size 4 \
--eval_epochs 5 \
--snapshot_epochs 5 \
--gpu_ids 0

