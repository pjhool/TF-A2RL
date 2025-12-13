#!/bin/bash
# A2RL Batch Processing Example Script
# This script demonstrates various ways to use A2RL_Batch.py

echo "======================================"
echo "A2RL Batch Processing Examples"
echo "======================================"
echo ""

# Example 1: Single Image Processing
echo "[Example 1] Processing single image..."
python A2RL_Batch.py --mode single \
    --image_path test_images/3846.jpg \
    --save_path test_images/3846_cropped.jpg

echo ""
echo "======================================"
echo ""

# Example 2: Batch Processing (Multiple Specific Images)
echo "[Example 2] Processing batch of specific images..."
python A2RL_Batch.py --mode batch \
    --image_paths test_images/3846.jpg test_images/1227.jpg test_images/1644.jpg \
    --output_dir batch_results/

echo ""
echo "======================================"
echo ""

# Example 3: Directory Processing (Recommended for Many Images)
echo "[Example 3] Processing entire directory with batch_size=8..."
python A2RL_Batch.py --mode directory \
    --input_dir test_images/ \
    --output_dir directory_results/ \
    --batch_size 8

echo ""
echo "======================================"
echo ""

# Example 4: Directory Processing with Larger Batch Size (for GPU)
echo "[Example 4] Processing directory with larger batch_size=16..."
python A2RL_Batch.py --mode directory \
    --input_dir test_images/ \
    --output_dir directory_results_large_batch/ \
    --batch_size 16

echo ""
echo "======================================"
echo ""

# Example 5: Processing Only Specific Extensions
echo "[Example 5] Processing only JPG files..."
python A2RL_Batch.py --mode directory \
    --input_dir test_images/ \
    --output_dir jpg_only_results/ \
    --batch_size 8 \
    --extensions jpg jpeg

echo ""
echo "======================================"
echo ""

# Example 6: Verbose Mode
echo "[Example 6] Processing with verbose output..."
python A2RL_Batch.py --mode directory \
    --input_dir test_images/ \
    --output_dir verbose_results/ \
    --batch_size 8 \
    --verbose

echo ""
echo "======================================"
echo "All examples completed!"
echo "======================================"
