# LCNet
if [ "$1" == 'l' ]; then	
    CUDA_VISIBLE_DEVICES=0 python main_stage1.py --in_img_num $2
fi

# NENet
if [ "$1" == 'n' ]; then	
    CUDA_VISIBLE_DEVICES=0 python main_stage2.py --in_img_num $2 --retrain $3
fi