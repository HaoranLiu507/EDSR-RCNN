# Train
 python main.py --model EDSR_RCNN --scale 2 --data_range 1-790/791-800 --save Experiment --n_colors 3 --n_resblocks 32 --res_scale 0.1 --loss 1*SmoothL1Loss --decay 100-200 --n_threads 24 --seed 7  --lr 1e-4 --RCNN_channel "on" --cuda "cuda"


# Test
# python main.py --data_test DIV2K --scale 2 --data_range 801-900 --pre_train 'Your pretrained model path' --test_only --n_colors 3 --n_resblocks 32 --res_scale 0.1 --loss 1*SmoothL1Loss --decay 100-200 --n_threads 24 --seed 7  --lr 1e-4 --save Experiment