#!/usr/bin/env python3
"""
Batch Video Frame Extractor for AI Dataset Creation
Processes multiple videos from a folder and extracts frames at specified FPS
"""

import cv2
import os
import sys
from pathlib import Path
import argparse
from datetime import datetime


def extract_frames_from_video(video_path, output_dir, target_fps=60, prefix="frame"):
    """
    Extract frames from a single video at specified FPS
    
    Args:
        video_path: Path to input video file
        output_dir: Directory to save extracted frames
        target_fps: Target frames per second to extract (default: 60)
        prefix: Prefix for output frame filenames
    
    Returns:
        Number of frames extracted, or 0 if failed
    """
    
    # Verify video file exists
    if not os.path.exists(video_path):
        print(f"  ✗ Error: Video file '{video_path}' not found!")
        return 0
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"  ✗ Error: Could not open video file '{video_path}'")
        return 0
    
    # Get video properties
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / original_fps if original_fps > 0 else 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"  Resolution: {width}x{height}")
    print(f"  Original FPS: {original_fps:.2f}")
    print(f"  Duration: {duration:.2f}s")
    
    # Calculate frame interval
    if target_fps >= original_fps:
        frame_interval = 1
        actual_fps = original_fps
    else:
        frame_interval = int(original_fps / target_fps)
        actual_fps = original_fps / frame_interval
    
    print(f"  Extracting at: {actual_fps:.2f} FPS")
    
    # Extract frames
    frame_count = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Save frame if it matches our interval
        if frame_count % frame_interval == 0:
            # Generate filename with zero-padded frame number
            filename = f"{prefix}_{saved_count:06d}.jpg"
            filepath = os.path.join(output_dir, filename)
            
            # Save frame
            cv2.imwrite(filepath, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved_count += 1
        
        frame_count += 1
    
    # Release video capture
    cap.release()
    
    print(f"  ✓ Extracted {saved_count} frames")
    
    return saved_count


def get_video_files(folder_path, extensions=None):
    """
    Get all video files from a folder
    
    Args:
        folder_path: Path to folder containing videos
        extensions: List of video file extensions (default: common formats)
    
    Returns:
        List of video file paths
    """
    
    if extensions is None:
        extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.m4v', '.webm']
    
    video_files = []
    
    for ext in extensions:
        video_files.extend(Path(folder_path).glob(f'*{ext}'))
        video_files.extend(Path(folder_path).glob(f'*{ext.upper()}'))
    
    return sorted(video_files)


def process_batch(input_folder, output_base_dir, target_fps=60, create_subfolders=True):
    """
    Process all videos in a folder
    
    Args:
        input_folder: Folder containing video files
        output_base_dir: Base directory for output
        target_fps: Target frames per second
        create_subfolders: Create separate subfolder for each video
    
    Returns:
        Dictionary with processing statistics
    """
    
    print("\n" + "="*70)
    print("BATCH VIDEO FRAME EXTRACTOR")
    print("="*70 + "\n")
    
    # Get all video files
    video_files = get_video_files(input_folder)
    
    if not video_files:
        print(f"✗ No video files found in: {input_folder}")
        return None
    
    print(f"Found {len(video_files)} video file(s) to process")
    print(f"Input folder: {input_folder}")
    print(f"Output folder: {output_base_dir}")
    print(f"Target FPS: {target_fps}")
    print(f"Subfolders: {'Yes' if create_subfolders else 'No'}")
    print("-"*70 + "\n")
    
    # Statistics
    stats = {
        'total_videos': len(video_files),
        'successful': 0,
        'failed': 0,
        'total_frames': 0,
        'videos_processed': []
    }
    
    # Process each video
    for idx, video_path in enumerate(video_files, 1):
        video_name = video_path.stem  # filename without extension
        
        print(f"[{idx}/{len(video_files)}] Processing: {video_path.name}")
        
        # Determine output directory and prefix
        if create_subfolders:
            output_dir = os.path.join(output_base_dir, video_name)
            prefix = "frame"
        else:
            output_dir = output_base_dir
            # Add video name to prefix when not using subfolders
            prefix = f"{video_name}_frame"
        
        # Extract frames
        frames_extracted = extract_frames_from_video(
            str(video_path),
            output_dir,
            target_fps,
            prefix
        )
        
        if frames_extracted > 0:
            stats['successful'] += 1
            stats['total_frames'] += frames_extracted
            stats['videos_processed'].append({
                'name': video_path.name,
                'frames': frames_extracted,
                'output': output_dir
            })
        else:
            stats['failed'] += 1
        
        print()  # Empty line between videos
    
    return stats


def print_summary(stats, start_time):
    """Print processing summary"""
    
    if not stats:
        return
    
    elapsed = datetime.now() - start_time
    
    print("="*70)
    print("PROCESSING SUMMARY")
    print("="*70)
    print(f"Total videos found: {stats['total_videos']}")
    print(f"Successfully processed: {stats['successful']}")
    print(f"Failed: {stats['failed']}")
    print(f"Total frames extracted: {stats['total_frames']:,}")
    print(f"Processing time: {elapsed.total_seconds():.2f} seconds")
    print("-"*70)
    
    if stats['videos_processed']:
        print("\nProcessed Videos:")
        for video in stats['videos_processed']:
            print(f"  • {video['name']}: {video['frames']:,} frames → {video['output']}")
    
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Batch extract frames from multiple videos for AI dataset creation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all videos in a folder
  python batch_extract_frames.py videos_folder
  
  # Specify output directory and FPS
  python batch_extract_frames.py videos_folder -o dataset_frames -f 30
  
  # Process single video file
  python batch_extract_frames.py video.mp4
  
  # Save all frames to one folder (no subfolders)
  python batch_extract_frames.py videos_folder -o frames --no-subfolders
        """
    )
    
    parser.add_argument('input', help='Input folder containing videos or single video file')
    parser.add_argument('-o', '--output', default='extracted_frames',
                       help='Output base directory (default: extracted_frames)')
    parser.add_argument('-f', '--fps', type=int, default=60,
                       help='Target FPS for extraction (default: 60)')
    parser.add_argument('--no-subfolders', action='store_true',
                       help='Save all frames to output folder without creating subfolders per video')
    
    args = parser.parse_args()
    
    start_time = datetime.now()
    
    # Check if input is a file or folder
    input_path = Path(args.input)
    
    if input_path.is_file():
        # Single file processing
        print("\n" + "="*70)
        print("SINGLE VIDEO FRAME EXTRACTOR")
        print("="*70 + "\n")
        
        video_name = input_path.stem
        output_dir = os.path.join(args.output, video_name)
        
        print(f"Input video: {input_path.name}")
        print(f"Output directory: {output_dir}")
        print(f"Target FPS: {args.fps}")
        print("-"*70 + "\n")
        
        frames = extract_frames_from_video(
            str(input_path),
            output_dir,
            args.fps
        )
        
        print("\n" + "="*70)
        if frames > 0:
            print(f"✓ Successfully extracted {frames:,} frames")
        else:
            print("✗ Frame extraction failed")
        print("="*70 + "\n")
        
    elif input_path.is_dir():
        # Batch processing
        stats = process_batch(
            str(input_path),
            args.output,
            args.fps,
            create_subfolders=not args.no_subfolders
        )
        
        if stats:
            print_summary(stats, start_time)
        
    else:
        print(f"Error: '{args.input}' is not a valid file or directory")
        sys.exit(1)


if __name__ == "__main__":
    main()