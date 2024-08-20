# EDSR baseline model (x2) + JPEG augmentation
#python main.py --model EDSR --scale 2 --patch_size 96 --save edsr_baseline_x2 --reset
#python main.py --model EDSR --scale 2 --patch_size 96 --save edsr_baseline_x2 --reset --data_train DIV2K+DIV2K-Q75 --data_test DIV2K+DIV2K-Q75

# EDSR baseline model (x3) - from EDSR baseline model (x2)
#python main.py --model EDSR --scale 3 --patch_size 144 --save edsr_baseline_x3 --reset --pre_train [pre-trained EDSR_baseline_x2 model dir]

# EDSR baseline model (x4) - from EDSR baseline model (x2)
#python main.py --model EDSR --scale 4 --save edsr_baseline_x4 --reset --pre_train [pre-trained EDSR_baseline_x2 model dir]

# EDSR in the paper (x2)
python main.py --model EDSR --scale 2 --data_range 1-790/791-800 --save EDSRTEST --n_resblocks 32 --res_scale 0.1 --loss 1*SmoothL1Loss --decay 100-200 --n_threads 24 --seed 7 --reset --lr 1e-4 --n_feats 100
# python main.py --model EDSR --scale 2 --data_range 1-790/791-800 --save 2023.07.06_DIV2Kgray_R32_RS0.1_F100_E300_LR1e-4_Gamma_0.5_SmoothL1Loss_PSNRc --n_colors 1 --n_resblocks 32 --res_scale 0.1  --loss 1*SmoothL1Loss --decay 100-200 --n_threads 24 --seed 7 --lr 1e-4 --n_feats 100 --pre_train '/home/6c702main/edsror/experiment/2023.07.06_DIV2Kgray_R32_RS0.1_F100_E300_LR1e-4_Gamma_0.5_SmoothL1Loss_PSNR/model/model_latest.pt'
# EDSR in the paper (x3) - from EDSR (x2)
#python main.py --model EDSR --scale 3 --save edsr_x3 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --reset --pre_train [pre-trained EDSR model dir]-

# EDSR in the paper (x4) - from EDSR (x2)
#python main.py --model EDSR --scale 4 --save edsr_x4 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --reset --pre_train [pre-trained EDSR_x2 model dir]

# MDSR baseline model
#python main.py --template MDSR --model MDSR --scale 2+3+4 --save MDSR_baseline --reset --save_models

# MDSR in the paper
#python main.py --template MDSR --model MDSR --scale 2+3+4 --n_resblocks 80 --save MDSR --reset --save_models
#python main.py --data_test CTDATA --data_range 790-795 --scale 2 --pre_train '/home/6c702main/EDSR-PyTorch-optuna/src/model_trial0.pt' --test_only --save_results --n_feats 419 --n_colors 1 --n_resblocks 24 --res_scale 0.38284538744039376
# Standard benchmarks (Ex. EDSR_baseline_x4)
#python main.py --data_test Set5+Set14+B100+Urban100+DIV2K --data_range 801-900 --scale 4 --pre_train download --test_only --self_ensemble

#python main.py --data_test Set5+Set14+B100+Urban100+DIV2K --data_range 801-900 --scale 4 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --pre_train download --test_only --self_ensemble

# Test your own images
#python main.py --data_test DIV2K --scale 2 --pre_train '/media/haoranliu/Project/Python Projects - Ubuntu/SuperResCT/model_validation/model_latest_T26.pt' --test_only --save_results --n_feats 300 --n_colors 3 --n_resblocks 30 --res_scale 0.3 --RCNN_channel "on"
 # Advanced - Test with JPEG images
#python main.py --model MDSR --data_test Demo --scale 2+3+4 --pre_train download --test_only --save_results

# Advanced - Training with adversarial loss
#python main.py --template GAN --scale 4 --save edsr_gan --reset --patch_size 96 --loss 5*VGG54+0.15*GAN --pre_train download

# RDN BI model (x2)
#python3.6 main.py --scale 2 --save RDN_D16C8G64_BIx2 --model RDN --epochs 200 --batch_size 16 --data_range 801-805 --patch_size 64 --reset
# RDN BI model (x3)
#python3.6 main.py --scale 3 --save RDN_D16C8G64_BIx3 --model RDN --epochs 200 --batch_size 16 --data_range 801-805 --patch_size 96 --reset
# RDN BI model (x4)
#python3.6 main.py --scale 4 --save RDN_D16C8G64_BIx4 --model RDN --epochs 200 --batch_size 16 --data_range 801-805 --patch_size 128 --reset

# RCAN_BIX2_G10R20P48, input=48x48, output=96x96
# pretrained model can be downloaded from https://www.dropbox.com/s/mjbcqkd4nwhr6nu/models_ECCV2018RCAN.zip?dl=0
#python main.py --template RCAN --save RCAN_BIX2_G10R20P48 --scale 2 --reset --save_results --patch_size 96
# RCAN_BIX3_G10R20P48, input=48x48, output=144x144
#python main.py --template RCAN --save RCAN_BIX3_G10R20P48 --scale 3 --reset --save_results --patch_size 144 --pre_train ../experiment/model/RCAN_BIX2.pt
# RCAN_BIX4_G10R20P48, input=48x48, output=192x192
#python main.py --template RCAN --save RCAN_BIX4_G10R20P48 --scale 4 --reset --save_results --patch_size 192 --pre_train ../experiment/model/RCAN_BIX2.pt
# RCAN_BIX8_G10R20P48, input=48x48, output=384x384
#python main.py --template RCAN --save RCAN_BIX8_G10R20P48 --scale 8 --reset --save_results --patch_size 384 --pre_train ../experiment/model/RCAN_BIX2.pt
#python main.py --model ABLATION --save ABLATION_TEST --scale 2 --save_results --save_models --loss 1*L1 --epochs 200 --reset
#python main.py --model EDSR --save RCNN_RESULT --scale 2 --save_results --pre_train '/home/6c702main/model_best.pt' --test_only
#python main.py --template VDSR --save TE --scale 2 --test_only --pre_train '/home/6c702main/01.15.2024 Archieve/SuperResCT/EDSR-PyTorch-optuna/experiment/VDSR_EX3/model/model_latest.pt'
#python main.py --model MDSR --scale 2 --pre_train '/home/6c702main/01.15.2024 Archieve/SuperResCT/model_validation/model_01.09/MDSR.pt' --test_only --n_resblocks 80 --n_feats 64
#python main.py --model EDSR --save_models --save_results --scale 2 --save ABAB --pre_train '/home/6c702main/01.15.2024 Archieve/SuperResCT/EDSR-PyTorch-optuna/experiment/abedsr_Result2/model/model_latest.pt' --test_only
#python main.py --model EDSR --save_results --scale 2 --save RCNN_BEST --pre_train '/home/6c702main/01.15.2024 Archieve/SuperResCT/EDSR-PyTorch-optuna/experiment/EDSR_RCNN_RESULT/model/model_best.pt' --test_only --RCNN_channel 'on'