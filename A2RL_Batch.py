from __future__ import absolute_import
import pickle
import argparse
import numpy as np
import tensorflow as tf
import skimage.io as io
import glob
import os
from tqdm import tqdm

import network
from actions import command2action, generate_bbox, crop_input

global_dtype = tf.float32

# Load pre-trained model
print("Loading pre-trained model...")
with open('vfn_rl.pkl', 'rb') as f:
    var_dict = pickle.load(f)

# Build TensorFlow graph
image_placeholder = tf.placeholder(dtype=global_dtype, shape=[None, 227, 227, 3])
global_feature_placeholder = network.vfn_rl(image_placeholder, var_dict)

h_placeholder = tf.placeholder(dtype=global_dtype, shape=[None, 1024])
c_placeholder = tf.placeholder(dtype=global_dtype, shape=[None, 1024])
action, h, c = network.vfn_rl(image_placeholder, var_dict, 
                              global_feature=global_feature_placeholder,
                              h=h_placeholder, c=c_placeholder)

# Create TensorFlow session
sess = tf.Session()
print("Model loaded successfully!")


def auto_cropping(origin_image, verbose=False):
    """
    Perform automatic cropping on a batch of images.
    
    Args:
        origin_image: List of numpy arrays (images)
        verbose: Print step information
    
    Returns:
        List of bounding boxes (xmin, ymin, xmax, ymax) for each image
    """
    batch_size = len(origin_image)
    
    if verbose:
        print(f"Processing batch of {batch_size} images...")
    
    # Initialize states
    terminals = np.zeros(batch_size)
    ratios = np.repeat([[0, 0, 20, 20]], batch_size, axis=0)
    img = crop_input(origin_image, generate_bbox(origin_image, ratios))
    
    # Extract global features (computed once per image)
    global_feature = sess.run(global_feature_placeholder, feed_dict={image_placeholder: img})
    
    # Initialize LSTM states
    h_np = np.zeros([batch_size, 1024])
    c_np = np.zeros([batch_size, 1024])
    
    step = 0
    # Iterative cropping loop
    while True:
        step += 1
        
        # Get action from RL network
        action_np, h_np, c_np = sess.run((action, h, c), 
                                         feed_dict={image_placeholder: img,
                                                   global_feature_placeholder: global_feature,
                                                   h_placeholder: h_np,
                                                   c_placeholder: c_np})
        
        # Convert action to bounding box update
        ratios, terminals = command2action(action_np, ratios, terminals)
        bbox = generate_bbox(origin_image, ratios)
        
        # Check if all images are done
        if np.sum(terminals) == batch_size:
            if verbose:
                print(f"Completed in {step} steps")
            return bbox
        
        # Crop images for next iteration
        img = crop_input(origin_image, bbox)


def process_single_image(image_path, save_path, verbose=True):
    """Process a single image."""
    if verbose:
        print(f"Processing: {image_path}")
    
    # Load image
    im = io.imread(image_path).astype(np.float32) / 255
    
    # Perform cropping
    xmin, ymin, xmax, ymax = auto_cropping([im - 0.5], verbose=verbose)[0]
    
    # Save result
    io.imsave(save_path, im[ymin:ymax, xmin:xmax])
    
    if verbose:
        print(f"Saved to: {save_path}")


def process_batch(image_paths, output_dir, verbose=True):
    """Process a batch of images."""
    if verbose:
        print(f"\nProcessing batch of {len(image_paths)} images...")
    
    # Load all images
    images = []
    original_images = []
    for path in image_paths:
        im = io.imread(path).astype(np.float32) / 255
        original_images.append(im)
        images.append(im - 0.5)
    
    # Perform batch cropping
    bboxes = auto_cropping(images, verbose=verbose)
    
    # Save results
    for path, bbox, orig_im in zip(image_paths, bboxes, original_images):
        xmin, ymin, xmax, ymax = bbox
        filename = os.path.basename(path)
        save_path = os.path.join(output_dir, f'cropped_{filename}')
        io.imsave(save_path, orig_im[ymin:ymax, xmin:xmax])
        
        if verbose:
            print(f"  Saved: {save_path}")


def process_directory(input_dir, output_dir, batch_size=8, extensions=None, verbose=True):
    """Process all images in a directory with batch processing."""
    if extensions is None:
        extensions = ['jpg', 'jpeg', 'png', 'bmp']
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect all image paths
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(input_dir, f'*.{ext}')))
        image_paths.extend(glob.glob(os.path.join(input_dir, f'*.{ext.upper()}')))
    
    if len(image_paths) == 0:
        print(f"No images found in {input_dir}")
        return
    
    print(f"Found {len(image_paths)} images in {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Batch size: {batch_size}")
    
    # Process in batches
    num_batches = (len(image_paths) + batch_size - 1) // batch_size
    
    for i in tqdm(range(0, len(image_paths), batch_size), 
                  desc="Processing batches", 
                  total=num_batches):
        batch_paths = image_paths[i:i+batch_size]
        process_batch(batch_paths, output_dir, verbose=False)
    
    print(f"\nAll {len(image_paths)} images processed successfully!")
    print(f"Results saved to: {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='A2RL: Batch Automatic Image Cropping',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single image
  python A2RL_Batch.py --mode single --image_path test.jpg --save_path output.jpg
  
  # Multiple images
  python A2RL_Batch.py --mode batch --image_paths img1.jpg img2.jpg img3.jpg --output_dir ./results/
  
  # Directory (recommended for many images)
  python A2RL_Batch.py --mode directory --input_dir ./test_images/ --output_dir ./results/ --batch_size 16
        """)
    
    parser.add_argument('--mode', choices=['single', 'batch', 'directory'], required=True,
                       help='Processing mode: single image, batch of images, or entire directory')
    
    # Single image mode
    parser.add_argument('--image_path', help='Path to single image (for single mode)')
    parser.add_argument('--save_path', help='Save path for single image (for single mode)')
    
    # Batch mode
    parser.add_argument('--image_paths', nargs='+', help='List of image paths (for batch mode)')
    
    # Directory mode
    parser.add_argument('--input_dir', help='Input directory (for directory mode)')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for processing (default: 8)')
    parser.add_argument('--extensions', nargs='+', default=['jpg', 'jpeg', 'png', 'bmp'],
                       help='Image extensions to process (default: jpg jpeg png bmp)')
    
    # Common
    parser.add_argument('--output_dir', help='Output directory (for batch/directory mode)')
    parser.add_argument('--verbose', action='store_true', help='Print detailed information')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.mode == 'single':
        if not args.image_path or not args.save_path:
            parser.error("--image_path and --save_path are required for single mode")
        process_single_image(args.image_path, args.save_path, verbose=args.verbose)
    
    elif args.mode == 'batch':
        if not args.image_paths or not args.output_dir:
            parser.error("--image_paths and --output_dir are required for batch mode")
        os.makedirs(args.output_dir, exist_ok=True)
        process_batch(args.image_paths, args.output_dir, verbose=args.verbose)
    
    elif args.mode == 'directory':
        if not args.input_dir or not args.output_dir:
            parser.error("--input_dir and --output_dir are required for directory mode")
        process_directory(args.input_dir, args.output_dir, 
                         batch_size=args.batch_size,
                         extensions=args.extensions,
                         verbose=args.verbose)
