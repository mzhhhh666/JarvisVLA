#! /bin/bash

cuda_visible_devices=0
card_num=1
model_name_or_path="/root/autodl-tmp/jarvisVLA-qwen2-vl-7B" #"/path/to/your/model/directory"

CUDA_VISIBLE_DEVICES=$cuda_visible_devices vllm serve $model_name_or_path \
    --port 9052 \
    --max-model-len 8448 \
    --max-num-seqs 10 \
    --gpu-memory-utilization 0.8 \
    --tensor-parallel-size $card_num \
    --trust-remote-code \
    --served_model_name "jarvisvla" \
    --limit-mm-per-prompt image=5 \
    #--dtype "float32" \
    #--kv-cache-dtype "fp8" \
