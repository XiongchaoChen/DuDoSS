python test.py \
--resume './outputs/train_1/checkpoints/model_199.pt' \
--experiment_name 'test_1_199' \
--model_type 'model_cnn' \
--data_root '../../Data/Processed_SV_15angle_dual2_08272021/' \
--net_G 'DuRDN' \
--norm 'BN' \
--DuRDN_filters 32 \
--batch_size 4 \
--eval_epochs 5 \
--snapshot_epochs 5 \
--gpu_ids 0


