#!/bin/bash

# ./run_server.sh {qwen3-4b-it | qwen3-30b-it | qwen3-30b-it-thinking}

MODEL_ID=${1}
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
HOST="127.0.0.1"
PORT=8100
DTYPE="auto"
TENSOR_PARALLEL_SIZE=1
ENABLE_PREFIX_CACHING=true

MODEL_DIR_BASE="${SAE_LLM_DIR:?Set SAE_LLM_DIR to your local language-model directory}"
case $MODEL_ID in
    qwen3-4b-it)
        MODEL_PATH="$MODEL_DIR_BASE/Qwen3-4B-Instruct-2507"
        GPU_MEMORY_UTILIZATION=0.9
        MAX_MODEL_LEN=32768
        MAX_NUM_SEQS=64
        ;;
    qwen3-30b-it)
        MODEL_PATH="$MODEL_DIR_BASE/Qwen3-30B-A3B-Instruct-2507"
        GPU_MEMORY_UTILIZATION=0.9
        MAX_MODEL_LEN=32768
        MAX_NUM_SEQS=16
        ;;
    qwen3-30b-it-thinking)
        MODEL_PATH="$MODEL_DIR_BASE/Qwen3-30B-A3B-Thinking-2507"
        GPU_MEMORY_UTILIZATION=0.9
        MAX_MODEL_LEN=32768
        MAX_NUM_SEQS=16
        ;;
    *)
        echo "Error: Unknown model ID '$MODEL_ID'"
        echo ""
        echo "Usage: $0 {qwen3-4b-it|qwen3-30b-it|qwen3-30b-it-thinking}"
        echo ""
        echo "Available models:"
        echo "  qwen3-4b-it              - 4B Instruct (fast)"
        echo "  qwen3-30b-it             - 30B Instruct (high quality)"
        echo "  qwen3-30b-it-thinking    - 30B Thinking (reasoning)"
        exit 1
        ;;
esac

echo "======================================"
echo "Starting vLLM Server"
echo "======================================"
echo "Model ID: ${MODEL_ID}"
echo "Model Path: ${MODEL_PATH}"
echo "URL: ${HOST}:${PORT}"
echo "GPU Memory: ${GPU_MEMORY_UTILIZATION}"
echo "Max Length: ${MAX_MODEL_LEN}"
echo "Max Seqs: ${MAX_NUM_SEQS}"
echo "======================================"

python -m vllm.entrypoints.openai.api_server \
    --served-model-name $MODEL_ID \
    --model $MODEL_PATH \
    --host $HOST \
    --port $PORT \
    --generation-config vllm \
    --dtype $DTYPE \
    --max-model-len $MAX_MODEL_LEN \
    --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
    --max-num-seqs $MAX_NUM_SEQS \
    --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
    --enable-prefix-caching
