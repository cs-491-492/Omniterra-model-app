export CUDA_VISIBLE_DEVICES=0
NUM_GPUS=1
export PYTHONPATH=$PYTHONPATH:`pwd`
ckpt_path='./models/hrnetw32.pth'
config_path='baseline.hrnetw32'
img_path='/home/bozcomlekci/Downloads/img'
python inference.py --ckpt_path=${ckpt_path} --config_path=${config_path} --img_path=${img_path}