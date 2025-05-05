#!/bin/bash

# 配置
PYTHON_SCRIPT="./datagen/nerfstudio2webdataset.py"  # Python 脚本路径
INPUT_DIR="../waymo_ns"                          # 输入目录
OUTPUT_DIR="../waymo_webdataset"                 # 输出目录
GPUS="2,3"                                   # 指定 GPU
NPROC_PER_NODE=2                                 # 每个节点的进程数

# 重试执行 torchrun
while true; do
    echo "Starting torchrun with GPUs: ${GPUS}"
    CUDA_VISIBLE_DEVICES=${GPUS} torchrun --nnodes=1 --nproc-per-node=${NPROC_PER_NODE} \
        "${PYTHON_SCRIPT}" -i "${INPUT_DIR}" -o "${OUTPUT_DIR}" 2>&1 | tee output.log

    # 检查错误：是否出现 CUDA Out Of Memory 错误
    if grep -q "torch.cuda.OutOfMemoryError" output.log; then
        echo "CUDA Out of Memory Error detected. Retrying..."
        sleep 5  # 等待 5 秒后重试
    else
        echo "Script completed successfully!"
        break
    fi
done

