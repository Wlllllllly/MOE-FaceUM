
ROOT=../../
export PYTHONPATH=$ROOT:$PYTHONPATH
port=53000
num_gpus=1
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=${num_gpus} --master_port=${port}  inference.py --config_path test.yaml \
                    --model_path ckpts/checkpoint_rank0_iter_newest.pth.tar \
                    --image_path images/2025073.jpg \
                    --results_path results \
                    --gpu_num 0 \


