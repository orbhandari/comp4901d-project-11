#!/bin/bash
# Wrapper script for llama-cli that forces it to exit after generation
# This is a workaround for llama-cli versions that enter conversation mode

# Get all arguments
LLAMA_CLI="$1"
shift  # Remove first argument (llama-cli path)

# Run llama-cli in background
"$LLAMA_CLI" "$@" &
PID=$!

# Wait for process with timeout
TIMEOUT=300  # 5 minutes
ELAPSED=0
SLEEP_INTERVAL=0.1

while kill -0 $PID 2>/dev/null; do
    sleep $SLEEP_INTERVAL
    ELAPSED=$(echo "$ELAPSED + $SLEEP_INTERVAL" | bc)
    
    # Check if timeout exceeded
    if (( $(echo "$ELAPSED > $TIMEOUT" | bc -l) )); then
        echo "Timeout exceeded, killing process" >&2
        kill -9 $PID 2>/dev/null
        exit 124  # Timeout exit code
    fi
    
    # Check if process is waiting for input (heuristic: no CPU usage for 2 seconds)
    # This is a simple check - if process is idle, it might be waiting for input
    if (( $(echo "$ELAPSED > 2" | bc -l) )); then
        # Check CPU usage (this is a simplified check)
        # If we detect the process is idle and printing >, kill it
        :  # Placeholder for now
    fi
done

# Get exit code
wait $PID
EXIT_CODE=$?

exit $EXIT_CODE
