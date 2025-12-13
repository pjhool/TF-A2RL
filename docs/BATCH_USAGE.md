# A2RL Batch Processing - Usage Guide

## Overview

`A2RL_Batch.py` extends the original A2RL with efficient batch processing capabilities. It supports three modes:
- **Single**: Process one image at a time
- **Batch**: Process multiple specific images
- **Directory**: Process all images in a folder

## Installation

Ensure you have the required dependencies:
```bash
pip install numpy tensorflow scikit-image tqdm
```

## Usage Examples

### 1. Single Image Mode

Process a single image (same as original A2RL):

```bash
python A2RL_Batch.py --mode single \
    --image_path test_images/3846.jpg \
    --save_path results/3846_cropped.jpg
```

### 2. Batch Mode

Process multiple specific images at once:

```bash
python A2RL_Batch.py --mode batch \
    --image_paths test_images/img1.jpg test_images/img2.jpg test_images/img3.jpg \
    --output_dir results/
```

### 3. Directory Mode (Recommended)

Process all images in a directory with automatic batching:

```bash
python A2RL_Batch.py --mode directory \
    --input_dir test_images/ \
    --output_dir results/ \
    --batch_size 16
```

**With custom extensions:**
```bash
python A2RL_Batch.py --mode directory \
    --input_dir test_images/ \
    --output_dir results/ \
    --batch_size 8 \
    --extensions jpg png jpeg
```

**With verbose output:**
```bash
python A2RL_Batch.py --mode directory \
    --input_dir test_images/ \
    --output_dir results/ \
    --batch_size 16 \
    --verbose
```

## Command Line Arguments

### Mode Selection
- `--mode {single,batch,directory}` **(required)**: Processing mode

### Single Mode Arguments
- `--image_path PATH`: Input image path
- `--save_path PATH`: Output image path

### Batch Mode Arguments
- `--image_paths PATH [PATH ...]`: List of input image paths
- `--output_dir DIR`: Output directory

### Directory Mode Arguments
- `--input_dir DIR`: Input directory containing images
- `--output_dir DIR`: Output directory for cropped images
- `--batch_size N`: Number of images to process simultaneously (default: 8)
- `--extensions EXT [EXT ...]`: Image file extensions to process (default: jpg jpeg png bmp)

### Common Arguments
- `--verbose`: Print detailed processing information

## Performance Recommendations

### Batch Size Guidelines

Choose batch size based on your GPU memory:

| GPU Memory | Recommended Batch Size | Expected Speedup |
|------------|----------------------|------------------|
| 4GB | 4-8 | 2-3x |
| 6GB | 8-12 | 3-4x |
| 8GB | 12-16 | 3.5-4.5x |
| 12GB+ | 16-32 | 4-5x |

### Tips for Best Performance

1. **Use Directory Mode for Many Images**
   - Automatically handles batching
   - Shows progress bar
   - Most efficient for bulk processing

2. **Adjust Batch Size**
   - Start with default (8)
   - Increase if you have more GPU memory
   - Decrease if you get OOM (Out of Memory) errors

3. **GPU vs CPU**
   - GPU: Use larger batch sizes (16-32)
   - CPU: Use smaller batch sizes (2-4)

## Output Format

All cropped images are saved with the prefix `cropped_`:
```
input_dir/
  ├── image1.jpg
  ├── image2.jpg
  └── image3.jpg

output_dir/
  ├── cropped_image1.jpg
  ├── cropped_image2.jpg
  └── cropped_image3.jpg
```

## Examples

### Example 1: Quick Test (Single Image)
```bash
python A2RL_Batch.py --mode single \
    --image_path test_images/3846.jpg \
    --save_path test_output.jpg
```

### Example 2: Process Specific Images
```bash
python A2RL_Batch.py --mode batch \
    --image_paths photo1.jpg photo2.jpg photo3.jpg \
    --output_dir ./cropped/
```

### Example 3: Bulk Processing (100+ Images)
```bash
python A2RL_Batch.py --mode directory \
    --input_dir ./raw_photos/ \
    --output_dir ./cropped_photos/ \
    --batch_size 16
```

### Example 4: Process Only JPG Files
```bash
python A2RL_Batch.py --mode directory \
    --input_dir ./images/ \
    --output_dir ./results/ \
    --extensions jpg jpeg \
    --batch_size 12
```

## Troubleshooting

### Out of Memory Error
**Problem**: `ResourceExhaustedError: OOM when allocating tensor`

**Solution**: Reduce batch size
```bash
python A2RL_Batch.py --mode directory \
    --input_dir ./images/ \
    --output_dir ./results/ \
    --batch_size 4  # Reduced from default 8
```

### No Images Found
**Problem**: `No images found in <directory>`

**Solution**: Check extensions
```bash
python A2RL_Batch.py --mode directory \
    --input_dir ./images/ \
    --output_dir ./results/ \
    --extensions jpg png jpeg bmp tiff  # Add more extensions
```

### Slow Processing
**Problem**: Processing is slower than expected

**Solutions**:
1. Increase batch size (if you have GPU memory)
2. Use GPU instead of CPU
3. Use directory mode instead of processing images one by one

## Comparison: Original vs Batch

### Original A2RL.py
```bash
# Process 100 images
for img in *.jpg; do
    python A2RL.py --image_path $img --save_path cropped_$img
done
# Time: ~500 seconds (5s per image)
```

### A2RL_Batch.py
```bash
# Process 100 images
python A2RL_Batch.py --mode directory \
    --input_dir ./ \
    --output_dir ./cropped/ \
    --batch_size 16
# Time: ~150 seconds (1.5s per image)
# Speedup: 3.3x faster!
```

## Advanced Usage

### Custom Batch Processing Script

For more control, you can import and use the functions directly:

```python
from A2RL_Batch import process_batch, process_directory

# Process specific images
image_paths = ['img1.jpg', 'img2.jpg', 'img3.jpg']
process_batch(image_paths, output_dir='./results/', verbose=True)

# Process directory with custom settings
process_directory(
    input_dir='./photos/',
    output_dir='./cropped/',
    batch_size=16,
    extensions=['jpg', 'png'],
    verbose=True
)
```

## Notes

- The model (`vfn_rl.pkl`) must be in the current directory
- All images in a batch are processed independently
- Different images may take different numbers of steps to complete
- The batch waits for all images to finish before returning results
- Progress bar (tqdm) shows batch-level progress, not individual images

## Getting Help

View all available options:
```bash
python A2RL_Batch.py --help
```
