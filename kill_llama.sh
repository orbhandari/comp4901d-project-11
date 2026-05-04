#!/bin/bash
# Emergency script to kill all llama processes

echo "Killing all llama-cli processes..."
pkill -9 llama-cli

echo "Killing all llama processes..."
pkill -9 llama

echo "Killing all timeout processes..."
pkill -9 timeout

echo "Done! All llama processes should be killed."
echo ""
echo "Check if any are still running:"
ps aux | grep llama | grep -v grep
