#!/bin/bash

CONFIG_PATH="configs/msrvtt_retrieval/msrvtt_retrieval_vip_base_32.json"
LOG_FILE="logs/inference_$(date +'%Y-%m-%d_%H-%M-%S').log"

echo "🔹 Running inference with config: $CONFIG_PATH"
python main.py --config $CONFIG_PATH | tee "$LOG_FILE"
echo "✅ Inference completed. Logs saved to $LOG_FILE"
