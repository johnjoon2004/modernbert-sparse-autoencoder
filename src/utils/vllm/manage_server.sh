#!/bin/bash

BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
RUN_SCRIPT="$BASE_DIR/start_server.sh"

find_current_model() {
    latest_log=$(ls -t "$BASE_DIR"/vllm_*.log 2>/dev/null | head -1)
    if [ -n "$latest_log" ]; then
        basename "$latest_log" .log | sed 's/vllm_//'
    else
        echo "unknown"
    fi
}

case "$1" in
    start)
        MODEL_ID=${2:-qwen3-4b-it}  # default: qwen3-4b-it

        # check if already running
        if pgrep -f "vllm.entrypoints.openai.api_server" > /dev/null; then
            echo "⚠️  vLLM server is already running!"
            echo "   Current model: $(find_current_model)"
            echo "   Stop it first: ./manage_server.sh stop"
            exit 1
        fi
        
        LOG_FILE="$BASE_DIR/vllm_${MODEL_ID}.log"
        
        echo "Starting vLLM server: $MODEL_ID"
        echo "Log file: $LOG_FILE"
        echo ""
        
        # run in background
        nohup bash "$RUN_SCRIPT" "$MODEL_ID" > "$LOG_FILE" 2>&1 &
        
        PID=$!
        echo "======================================"
        echo "vLLM Server Started"
        echo "======================================"
        echo "PID: $PID"
        echo "Model: $MODEL_ID"
        echo "Log: $LOG_FILE"
        echo ""
        echo "Useful commands:"
        echo "  ./manage_server.sh status    - Check server status"
        echo "  ./manage_server.sh logs      - View logs"
        echo "  ./manage_server.sh stop      - Stop server"
        ;;        

    stop)
        echo "Stopping vLLM server..."
        PID=$(pgrep -f "vllm.entrypoints.openai.api_server")

        if [ -z "$PID" ]; then
            echo "vLLM server is not running."
            exit 0
        fi
    
        echo "Found PID: $PID"
        kill -TERM "$PID"
        echo "Sent SIGTERM, waiting for graceful shutdown..."
        
        # wait 15 seconds for graceful shutdown
        for i in {1..15}; do
            if ! kill -0 $PID 2>/dev/null; then
                echo ""
                echo "✅ vLLM server stopped gracefully."
                exit 0
            fi
            sleep 1
            echo -n "."
        done

        echo ""
        echo "Server not responding, force stopping..."
        kill -KILL "$PID" 2>/dev/null
        echo "✅ vLLM server force stopped."
        ;;

    status)
        if pgrep -f "vllm.entrypoints.openai.api_server" > /dev/null; then
            CURRENT_MODEL=$(find_current_model)
            PID=$(pgrep -f "vllm.entrypoints.openai.api_server")
            
            echo "✅ vLLM server is running"
            echo "   PID: $PID"
            echo "   Model: $CURRENT_MODEL"
            echo "   Port: 8100"
            echo ""
            
            GPU_MEM=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null | grep "^$PID" | awk '{print $2}')
            if [ -n "$GPU_MEM" ]; then
                echo "   GPU Memory: ${GPU_MEM} MB"
            else
                echo "   GPU Memory: (unable to query)"
            fi
        else
            echo "❌ vLLM server is not running."
            echo ""
            echo "Start server with:"
            echo "  ./manage_server.sh start [model_id]"
            echo ""
            echo "Available models:"
            echo "  qwen3-4b-it              - 4B Instruct (fast)"
            echo "  qwen3-30b-it             - 30B Instruct (high quality)"
            echo "  qwen3-30b-it-thinking    - 30B Thinking (reasoning)"
        fi
        ;;

    logs)
        CURRENT_MODEL=$(find_current_model)
        LOG_FILE="$BASE_DIR/vllm_${CURRENT_MODEL}.log"
        
        if [ -f "$LOG_FILE" ]; then
            echo "Showing logs for: $CURRENT_MODEL"
            echo "Log file: $LOG_FILE"
            echo "======================================"
            tail -f "$LOG_FILE"
        else
            echo "❌ No log file found."
            echo ""
            echo "Available log files:"
            ls -1 "$BASE_DIR"/vllm_*.log 2>/dev/null || echo "  (none)"
        fi
        ;;
        
    *)
        echo "Usage: $0 {start|stop|status|logs} [model_id]"
        echo ""
        echo "Commands:"
        echo "  start [model]   Start vLLM server (default: qwen3-4b-it)"
        echo "  stop            Stop vLLM server"
        echo "  status          Check server status and GPU usage"
        echo "  logs            Show logs of current model"
        echo ""
        echo "Available models:"
        echo "  qwen3-4b-it              - 4B Instruct (fast)"
        echo "  qwen3-30b-it             - 30B Instruct (high quality)"
        echo "  qwen3-30b-it-thinking    - 30B Thinking (reasoning)"
        echo ""
        echo "Examples:"
        echo "  $0 start                      # Start with default model (qwen3-4b-it)"
        echo "  $0 start qwen3-30b-it         # Start with 30B model"
        echo "  $0 status                     # Check server status"
        echo "  $0 logs                       # View logs (Ctrl+C to exit)"
        echo "  $0 stop                       # Stop server"
        exit 1
        ;;
esac