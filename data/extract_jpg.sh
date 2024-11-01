#!/bin/bash

# Directory containing MP4 files
VIDEOS_DIR=$1

# Directory where JPEG files will be saved
OUTPUT_DIR=$2

# Check if output directory exists, if not create it
[ ! -d "$OUTPUT_DIR" ] && mkdir -p "$OUTPUT_DIR"

# Loop through all MP4 files in the directory
for video in "$VIDEOS_DIR"/*.mp4; do
    # Extract filename without extension
    base_name=$(basename "$video" .mp4)
    
    # Command to extract frames every second and save as JPEG
    ffmpeg -i "$video" -vf "fps=30" "$OUTPUT_DIR/${base_name}_%04d.jpg"
done

echo "Frame extraction complete."

