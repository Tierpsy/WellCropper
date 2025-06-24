#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Video Cropping Tool for Multi-Well Plates 
IMPORTANT: This script should be run in a tierpsy environment with the necessary dependencies installed.

Main workflow:
1. Recursively find all video files in the input directory.
2. Group videos by plate.
3. For each video, crop wells and save metadata.
4. For each plate, construct a composite annotated image from the first frame of each channel.


Arguments:
    input_dir: Path to the input folder containing videos.
    param_file: Path to the JSON parameter file containing 'microns_per_pixel' and 'MWP_mapping'. It's compatible with
    parameter file used for batch processing.
    output_dir: Optional path to the output folder where cropped videos will be saved.
                If not provided, it defaults to a folder named "CroppedVideos" will be made in the parent directory of input_dir.
    threads: Number of threads for parallel processing (default is 1).
    log_level: Set the logging level (default is WARNING). Options are DEBUG, INFO, WARNING, ERROR, CRITICAL.


The script is designed to work with videos with a specific directory structure
and naming convention, where each videos are located in subfolders named as {plate_id + "." + Camera ID}.

Well names are read based on "mwp_mapping" from the JSON file, which contains the well layout information source.

Author: Hossein Khabbaz
Date: 2025-06-11
"""

import os
import argparse
import time
import cv2
import subprocess
import yaml
import shutil
import json
import logging
import numpy as np
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from tierpsy import DFLT_SPLITFOV_PARAMS_PATH
from tierpsy.helper.params.tracker_param import SplitFOVParams
from tierpsy.analysis.split_fov.FOVMultiWellsSplitter import FOVMultiWellsSplitter

# Start timing the script
start_time = time.time()

def represent_list_flow(self, data):
    """
    Custom YAML representer for lists to ensure they are represented in flow style.
    """
    return self.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)

yaml.add_representer(list, represent_list_flow)

def find_video_files(input_dir, extensions=(".mp4", ".avi", ".mov")):
    """Recursively yield video file paths from a directory."""
    for dirpath, _, files in os.walk(input_dir):
        for f in files:
            if f.lower().endswith(extensions):
                yield Path(dirpath) / f

def group_videos_by_plate_and_channel(video_files):
    """
    Groups video files by plate ID (folder_name up to ".") and assigns channel numbers by order.
    Args:
        video_files: List of video file paths.
    Returns:
        plate_dict: Dictionary where keys are plate IDs and values are list of corresponding video paths.
        Example: 
        {
            'Plate1': ['path/to/video1.mp4','path/to/video2.mp4', ...],
            'Plate2': ['path/to/video3.mp4', ...],
            ...
        }
    """
    # Group videos by the prefix of their parent folder (before the dot), which is used as plate_id.
    from collections import defaultdict
    plate_dict = defaultdict(list)
    for video_path in video_files:
        dir_stem = video_path.parent.stem
        plate_id = dir_stem.rsplit('.', 1)[0]
        plate_dict[plate_id].append(video_path)
    return plate_dict

def read_coordinates(video_path, param_file, img_fov):
    """
    Read and extract crop regions from the well configuration.
    Args:
        video_path: Absolute path to the video file.
        param_file: Path to the JSON parameter file containing 'microns_per_pixel' and 'MWP_mapping'.
        img_fov: First frame of the video, used for well detection.
    Returns:
        crop_specs: List of (x_min, y_min, width, height, well_name) for each well that will be cropped and annotated later.
        wells_df: DataFrame with well centers and radii.
        shape, ch: Metadata for the video.
    """
    try:
        with open(param_file, 'r') as fid:
            param_data = json.load(fid)

        # Extract parameters with defaults
        microns_per_pixel = float(param_data.get('microns_per_pixel', 12.4))
        mapping_filename = param_data['MWP_mapping']
        mapping_path = Path(DFLT_SPLITFOV_PARAMS_PATH) / mapping_filename

        # Load mapping file via SplitFOVParams
        splitfovparams = SplitFOVParams(json_file=str(mapping_path))
        shape, edge_frac, sz_mm = splitfovparams.get_common_params()
        uid, rig, ch, mwp_map = splitfovparams.get_params_from_filename(video_path)

        # Run FOV splitter
        fovsplitter = FOVMultiWellsSplitter(
            img_fov,
            microns_per_pixel=microns_per_pixel,
            well_shape=shape,
            well_size_mm=sz_mm,
            well_masked_edge=edge_frac,
            camera_serial=uid,
            rig=rig,
            channel=ch,
            wells_map=mwp_map
        )
        wells = fovsplitter.get_wells_data()
        wells_df = fovsplitter.wells[['x', 'y', 'r']]


        df = wells[wells['is_good_well'] == 1]
        crop_specs = []

        for _, row in df.iterrows():
            x_min, x_max = int(row['x_min']), int(row['x_max'])
            y_min, y_max = int(row['y_min']), int(row['y_max'])
            width, height = x_max - x_min, y_max - y_min
            well_name = str(row['well_name']).replace("b'", "").replace("'", "")
            crop_specs.append((x_min, y_min, width, height, well_name))

        return crop_specs, wells_df, shape, ch, rig
    except Exception as e:
        logging.error(f"Error reading coordinates: {e}")
        return []

def putText_upside_down(img, well_name, org, font, font_scale, color, thickness):
    """ 
    Draw text upside down on an image.
    Used for channels where the frame will be rotated after annotation,
    so the text appears upright in the final composite image.
    Args:
        img: The image to draw on.
        text: The text to draw.
        org: The bottom-left corner of the text (x, y).
        font: The font type (e.g., cv2.FONT_HERSHEY_SIMPLEX).
        font_scale: The scale of the font.
        color: The color of the text in BGR format.
        thickness: The thickness of the text.
    Returns:
        img: The image with the upside-down text drawn on it.
        """
    # Create a blank image for the text
    (w, h), baseline = cv2.getTextSize(well_name, font, font_scale, thickness)
    text_img = np.zeros((h + baseline, w, 3), dtype=np.uint8)
    # Draw the text (upright) on the blank image
    cv2.putText(text_img, well_name, (0, h), font, font_scale, color, thickness)
    # Rotate the text image 180 degrees
    text_img = cv2.rotate(text_img, cv2.ROTATE_180)
    # Overlay the rotated text image onto the main image at org
    x, y = org
    y1, y2 = y, y + text_img.shape[0]
    x1, x2 = x, x + text_img.shape[1]
    # Check bounds
    if y1 < 0 or y2 > img.shape[0] or x1 < 0 or x2 > img.shape[1]:
        return img  # skip if out of bounds
    # Overlay
    mask = text_img.any(axis=2)
    img[y1:y2, x1:x2][mask] = text_img[mask]
    return img

def draw_well_overlays(frame, crop_specs, ch, color=(0, 255, 0), thickness=8):
    """
    Draw rectangles and well names for each well on the frame.
    For Ch1, Ch3, Ch5, well names are drawn upside down to compensate for later rotation.
    Args:
        frame: The image frame to draw on.
        crop_specs: List of tuples (x_min, y_min, width, height, well_name) for each well.
        ch: Channel name (e.g., 'Ch1', 'Ch2', etc.) to determine text orientation.
        color: Color of the rectangle and text in BGR format.
        thickness: Thickness of the rectangle border.
    Returns:
        overlay: An image with the same size as video with the well rectangles and names drawn on it.
    """ 
    overlay = frame.copy()
    for x_min, y_min, width, height, well_name in crop_specs:
        x_min, y_min, width, height = int(x_min), int(y_min), int(width), int(height)
        top_left = (x_min, y_min)
        bottom_right = (x_min + width, y_min + height)
        cv2.rectangle(overlay, top_left, bottom_right, color, thickness)
        # draw well name
        if ch in ['Ch1', 'Ch3', 'Ch5']:
            overlay = putText_upside_down(
                overlay, str(well_name), (x_min + 65, y_min + 250),
                cv2.FONT_HERSHEY_SIMPLEX, 10, color, 12
            )
        else:
            cv2.putText(
                overlay, str(well_name), (x_min + 65, y_min + 250),
                cv2.FONT_HERSHEY_SIMPLEX, 10, color, 12
            )  
    return overlay

def crop_video(input_file, output_file, x_min, y_min, width, height, well_name, encoder):
    """Crop a video using FFmpeg."""
    command = [
        'ffmpeg',
        '-i', input_file,
        '-filter:v', f'crop={width}:{height}:{x_min}:{y_min}',
        '-c:v', encoder, 
        '-preset', 'medium',
        '-c:a', 'copy',
        '-threads', '1',
        output_file,
        '-y'
    ]
    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        logging.info(f"[✓] Cropped well {well_name}: {output_file}")
        return True
    except subprocess.CalledProcessError as e:
        logging.warning(f"[X] Error cropping video {input_file} for well {well_name}")
        print(e.stderr.decode())
        return False

def copy_and_update_metadata(original_metadata_path, output_dir, new_shape, x_min, y_min, video_stem, well_name, well_data, well_shape):
    """
    Copy and update metadata.yaml for each cropped well. 
    If metadata.yaml does not exist, create a new one with the necessary fields.
    This function also handles copying and updating .npz and .extra_data.json files if they exist.

    Args:
        original_metadata_path (Path): Path to the original metadata.yaml file.
        output_dir (Path): Target directory to save updated metadata.yaml.
        new_shape (tuple): (height, width) of the cropped video.
        x_min (int): x-coordinate of the top-left corner of the crop.
        y_min (int): y-coordinate of the top-left corner of the crop.
        video_stem (str): Stem of the video file name, used to construct output file names.
        well_name (str): Name of the well being processed.
        well_data (DataFrame row): Data for the well, containing 'x', 'y', and 'r' (radius) if applicable.
        well_shape (str): Shape of the well ('circle' or 'rectangle').
    Outputs:
        Writes metadata.yaml to output_dir with updated fields.
        Copies and updates .npz and .extra_data.json files if they exist.
    """
    output_metadata_path = output_dir / 'metadata.yaml'

    try:
        if original_metadata_path.exists():
            with open(original_metadata_path, 'r') as f:
                metadata = yaml.safe_load(f) or {}
        else:
            logging.info(f"[!] metadata.yaml not found at {original_metadata_path}, creating new one.")
            metadata = {}

        # Ensure the __store section exists
        if '__store' not in metadata or not isinstance(metadata['__store'], dict):
            metadata['__store'] = {}

        # Update shape and offset
        metadata['__store']['imgshape'] = list(new_shape)  # [height, width]
        metadata['__store']['imgoffset'] = [x_min, y_min]
        metadata['__store']['well_name'] = well_name 
        metadata['__store']['well_centre'] = [int(well_data['x']), int(well_data['y'])]
        metadata['__store']['well_shape'] = well_shape
        # add well_radius only if the well shape is a circle
        if well_shape == 'circle':
            metadata['__store']['well_radius'] = int(well_data['r'])
        

        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_metadata_path, 'w') as f:
            yaml.dump(metadata, f)

        logging.info(f"[✓] metadata.yaml written to {output_metadata_path}")
    except Exception as e:
        logging.warning(f"[X] Failed to write metadata.yaml: {e}")
    
    # Handle .npz file
    npz_path = original_metadata_path.parent / (video_stem + '.npz')
    if npz_path.exists():
        try:
            npz_data = np.load(npz_path, allow_pickle=True)
            npz_dict = dict(npz_data)
            npz_dict['imgshape'] = np.array(new_shape)
            output_npz_path = output_dir / npz_path.name
            np.savez(output_npz_path, **npz_dict)
            logging.info(f"[✓] {npz_path.name} written to {output_npz_path}")
        except Exception as e:
            logging.warning(f"[X] Failed to update/copy {npz_path.name}: {e}")

    # Handle .extra_data.json file
    extra_json_path = original_metadata_path.parent / (video_stem + '.extra_data.json')
    if extra_json_path.exists():
        try:
            output_json_path = output_dir / extra_json_path.name
            shutil.copy2(extra_json_path, output_json_path)
            logging.info(f"[✓] {extra_json_path.name} copied to {output_json_path}")
        except Exception as e:
            logging.warning(f"[X] Failed to copy {extra_json_path.name}: {e}")


def process_video(video_path, input_root , output_root, param_file, encoder):
    """
    Handles full croping process of a single video including reading coordinates, annotating the frame,
    cropping the video, and saving metadata.
    Args: 
        video_path: Path to the video file
        input_root: The root directory where the script will recursively search for video files to crop.
        output_root: The base directory where all cropped well videos (and their metadata) will be saved.
        param_fil: Path to the JSON parameter file containing microns_per_pixel and MWP_mapping. 
    Outputs:
        success_count: Number of wells successfully cropped from the video.
        ch: Channel name of the video (e.g., 'Ch1', 'Ch2', etc.).
        img_annot: Annotated frame with well rectangles and names drawn on it.
        plate_side_width: Width of the plate side area, calculated from the crop specifications.
        rig: The rig identifier extracted from the video metadata.
    """
    # Initialize variables
    plate_side_width = None
    try:
        rel_path = video_path.relative_to(input_root)
        video_dir = video_path.parent
        video_stem = video_path.stem

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logging.warning(f"[X] Failed to open: {video_path}")
            return 0, None, None, None, None
        
        # Read the first frame to get the image size and crop specifications
        ret, img_fov = cap.read()
        if not ret:
            logging.warning(f"[X] Failed to read frame from: {video_path}")
            return 0, None, None, None, None

        # extract crop specifications and metadata
        crop_specs, wells_df, shape, ch, rig= read_coordinates(str(video_path), str(param_file), img_fov)
        if not crop_specs:
            print(f"[!] No valid wells for: {video_path}")
            return 0, None, None, None, None
        
        # Draw rectangles for wells using crop_specs
        img_annot = draw_well_overlays(img_fov, crop_specs, ch)

        # find the plate side width from the crop_specs
        if ch == 'Ch1':
            wells_area_width = max(x_min + width for x_min, y_min, width, height, well_name in crop_specs)
            plate_side_width = img_annot.shape[1] - wells_area_width

        success_count = 0
        for (x_min, y_min, width, height, well_name), (well_index, well_data) in tqdm(zip(crop_specs, wells_df.iterrows()), desc=f"{video_stem} Processing wells", unit=" well"):
            # Output folder: output_root/<same_structure>/<videoname>_well_<well_name>/
            well_subdir = f"{video_stem}_well_{well_name}"
            target_dir = output_root / rel_path.parent / well_subdir
            target_dir.mkdir(parents=True, exist_ok=True)

            output_file = target_dir / "000000.mp4"
            if crop_video(str(video_path), str(output_file), x_min, y_min, width, height, well_name, encoder):
                success_count += 1

                # Check if metadata.yaml exists in the same folder as the video, update it, if not, create one 
                metadata_path = video_dir / 'metadata.yaml'
                copy_and_update_metadata(metadata_path, target_dir, (height, width), x_min, y_min, video_stem, well_name, well_data, shape) 


        return success_count, ch, img_annot, plate_side_width, rig

    except Exception as e:
        print(f"[X] Exception processing {video_path}: {e}")
        return 0, None, None, None, None

def construct_plate_image(plate_id, ch_frame_dict, plate_side_width, rig, output_root):
    """
    Construct a composite plate image from the first frame of each channel video.
    Arranges frames in a 2x3 grid: Ch1,Ch3,Ch5 (top), Ch2,Ch4,Ch6 (bottom).
    Resizes all frames to (max_h, max_w) so they fit perfectly.
    Saves the composite image to output_root/plate_id_plate_image.jpg

    Important:
    channel positioning is based on pos_map:
    {
        'Ch1': (0,0), 'Ch3': (0,1), 'Ch5': (0,2),
        'Ch2': (1,0), 'Ch4': (1,1), 'Ch6': (1,2)
    }

    args:
        plate_id: The ID of the plate (e.g., "Plate1"), video parent folder name before ".".
        ch_frame_dict: Dictionary containing channel frames {channel_name: frame}.
        plate_side_width: Width of the plate side area, used to cut the plate side from Ch1 frame.
        rig: The rig identifier extracted based on parameter file.
        output_root: The root directory where the composite image will be saved.
    outputs:
        Saves a composite image of the plate with well frames arranged in a grid.
    """
    # Define positions for each channel in the composite image
    pos_map = {'Ch1': (0,0), 'Ch3': (0,1), 'Ch5': (0,2), 'Ch2': (1,0), 'Ch4': (1,1), 'Ch6': (1,2)}
    
    # thickness of the border around each frame
    border = 50
    
    # Find max height and width among all frames
    sizes = [frame.shape[:2] for frame in ch_frame_dict.values() if frame is not None]
    if not sizes:
        print(f"[!] No valid frames for plate {plate_id}")
        return
    max_h = max(h for h, w in sizes)
    max_w = max(w for h, w in sizes)
    
    # Cut out plate side from wells part
    frame_ch1 = ch_frame_dict.get('Ch1')
    if frame_ch1 is not None:
        if plate_side_width > 0:
            # cut plate side from the wells part
            ch_frame_dict['Ch1'] = frame_ch1[:, :-1*plate_side_width] 
            
            # Rotate Ch1 frame 180 degrees to match the orientation in the final composite image
            frame_ch1_rotated = cv2.rotate(frame_ch1, cv2.ROTATE_180)
            plate_side = frame_ch1_rotated[:, :plate_side_width]
            
            # Resize plate side to match the height of the frames
            plate_side = cv2.resize(plate_side, (plate_side_width, max_h), interpolation=cv2.INTER_AREA)
            plate_side_with_border = cv2.copyMakeBorder(
                plate_side, border, border, border, border, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    else:
        logging.warning("Ch1 frame is missing.")

    # Create a blank image for the plate
    frame_h = max_h + 2 * border
    frame_w = max_w + 2 * border
    canvas_h = 2 * frame_h
    canvas_w = 3 * frame_w
    plate_img = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    # Place each channel frame in the correct position
    for ch, frame in ch_frame_dict.items():
        pos = pos_map.get(ch)
        if pos is None or frame is None:
            continue
        row, col = pos

        # Resize and add border as before
        frame_resized = cv2.resize(frame, (max_w, max_h), interpolation=cv2.INTER_AREA)
        if ch in ['Ch1', 'Ch3', 'Ch5']:
            frame_resized = cv2.rotate(frame_resized, cv2.ROTATE_180)
        frame_resized_with_border = cv2.copyMakeBorder(
            frame_resized, border, border, border, border, cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )

        # Calculate placement indices
        y_start = row * (frame_h)
        y_end = y_start + frame_h
        x_start = col * (frame_w)
        x_end = x_start + frame_w

        plate_img[y_start:y_end, x_start:x_end] = frame_resized_with_border

    # In case plate_side_width is provided, 
    # add the plate side to the left side of the plate image
    if ch_frame_dict.get('Ch1') is not None:
        h_side, w_side = plate_side_with_border.shape[:2]
        h_img = plate_img.shape[0]
        pad_height = h_img - h_side
        # Pad only at the bottom
        plate_side_padded = np.pad(
            plate_side_with_border,
            ((0, pad_height), (0, 0), (0, 0)),
            mode='constant',
            constant_values=0
        )
        # Concatenate plate side and plate image
        completed_plate_img = np.concatenate([plate_side_padded, plate_img], axis=1)
    else:
        completed_plate_img = plate_img
    # Resize to 25% of original width and height
    completed_plate_img_resized = cv2.resize(
        completed_plate_img,
        (completed_plate_img.shape[1] // 4, completed_plate_img.shape[0] // 4),
        interpolation=cv2.INTER_AREA
    )
    out_path = output_root / f"{plate_id}_{rig}_plate_image.jpg"
    cv2.imwrite(str(out_path), completed_plate_img_resized)
    print(f"[✓] Plate image saved: {out_path}")

def main():
    """
    Main entry point: parses arguments, finds videos, processes each plate and video,
    and constructs composite images.
    """
    parser = argparse.ArgumentParser(description="Crop wells from videos in a folder.")
    parser.add_argument("input_dir", help="Path to input folder containing videos")
    parser.add_argument("param_file", help="JSON parameter file with 'microns_per_pixel' and 'MWP_mapping'")
    parser.add_argument("--output-dir", help="Optional path to output folder")
    parser.add_argument("--threads", type=int, default=1, help="Number of threads for parallel processing")
    parser.add_argument("--encoder", default="cpu", choices=["cpu", "nvidia", "apple"], help="Video encoder: 'cpu' (libx264, default), 'nvidia' (h264_nvenc), or 'apple' (h264_videotoolbox)")
    parser.add_argument("--log-level", default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level (default: WARNING)"
    )
    args = parser.parse_args()
    encoder_map = {
        "cpu": "libx264",
        "nvidia": "h264_nvenc",
        "apple": "h264_videotoolbox"
        }
    encoder = encoder_map[args.encoder]

    # Set up logging
    log_level = getattr(logging, args.log_level.upper(), logging.WARNING)
    logging.basicConfig(
        level=log_level,
        format='[%(levelname)s] %(message)s',
        handlers=[
            logging.StreamHandler(),
        ]
    )

    # Resolve paths
    input_root = Path(args.input_dir).resolve()
    json_file = Path(args.param_file).resolve()
    output_root = Path(args.output_dir).resolve() if args.output_dir else input_root.parent / "CroppedVideos"

    print(f"Scanning videos in: {input_root}")
    print(f"Saving cropped videos to: {output_root}")

    video_files = list(find_video_files(input_root))
    if not video_files:
        logging.warning("No video files found.")
        return

    print(f"Found {len(video_files)} video(s). Starting cropping...")

    # Group by plate and channel
    plate_dict = group_videos_by_plate_and_channel(video_files)

    # Build a flat list of (plate_id, video_path)
    video_tasks = []
    for plate_id, videos in plate_dict.items():
        for video_path in videos:
            video_tasks.append((plate_id, video_path))
    

    # Process each video in parallel
    results = []
    total_wells = 0
    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        futures = [
            executor.submit(process_video, video_path, input_root, output_root, json_file, encoder)
            for plate_id, video_path in video_tasks
        ]
        for idx, f in enumerate(tqdm(futures, desc="Videos", unit="video")):
            success_count, ch, frame, plate_side_width, rig = f.result()
            total_wells += success_count
            # Find plate_id for this video
            plate_id = video_tasks[idx][0]
            results.append((plate_id, ch, frame, plate_side_width, rig))

    # Group results by plate_id
    plate_results = defaultdict(list)
    for plate_id, ch, frame, plate_side_width, rig in results:
        plate_results[plate_id].append((ch, frame, plate_side_width, rig))

    # Construct composite images for each plate
    print("Processing videos completed.")
    print("Constructing composite plate images...")
    for plate_id, ch_results in tqdm(plate_results.items(), desc="Plates", unit="plate"):
        ch_frame_dict = {}
        plate_side_width = None
        rig = None
        for ch, frame, this_plate_side_width, this_rig in ch_results:
            if ch is not None and frame is not None:
                ch_frame_dict[ch] = frame
            if ch == "Ch1" and this_plate_side_width:
                plate_side_width = this_plate_side_width
                rig = this_rig
        try:
            construct_plate_image(plate_id, ch_frame_dict, plate_side_width, rig, output_root)
        except Exception as e:
            logging.error(f"[X] Error constructing plate image for {plate_id}: {e}")

    print(f"Done. Total wells cropped: {total_wells}")
    print("--- %s seconds ---" % (time.time() - start_time))

if __name__ == "__main__":
    main()
