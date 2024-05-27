#!/bin/bash

# Check the first argument passed to the script
case "$1" in
    train)
        echo "Running training script..."
        # Call your training script here
        CUDA_VISIBLE_DEVICES=0 python training/dqn.py -train -s 10000
        ;;
    test)
        echo "Running testing script..."
        # Call your testing script here
        CUDA_VISIBLE_DEVICES=0 python training/dqn.py
        ;;
    *)
        echo "Error: Invalid argument. Use 'train' or 'test'."
        exit 1
        ;;
esac