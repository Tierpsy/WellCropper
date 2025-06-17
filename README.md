# WellCropper

**WellCropper** is a high-throughput video cropping and annotation tool for multi-well plate imaging, designed for use in a [Tierpsy](https://github.com/Tierpsy/tierpsy-tracker) environment.

This script automates the process of:
- Recursively finding all video files in a directory tree
- Grouping videos by plate and channel
- Cropping individual wells from each video using well layout metadata
- Saving cropped well videos and updated metadata
- Generating composite annotated images for each plate

**Features:**
- Supports 24, and 96-well plate layouts (customizable via JSON mapping)
- Parallel processing for fast batch cropping (user-configurable threads)
- Hardware-accelerated video encoding (Apple Silicon, NVIDIA, or CPU)
- Compatible with Tierpsy metadata and directory structure
- Generates composite images for easy QC and visualization

**Typical Use Case:**
- You have a directory of multi-well plate videos (e.g., from a high-content imaging rig)
- You want to extract each well as a separate video, with correct metadata, for downstream analysis in Tierpsy or other tools

## Usage

```sh
python wells_cropper.py <input_dir> <param_file> [--output-dir OUTPUT_DIR] [--threads N] [--encoder cpu|nvidia|apple] [--log-level LEVEL]
```

- `input_dir`: Path to the folder containing raw videos (searched recursively)
- `param_file`: JSON parameter file with `microns_per_pixel` and `MWP_mapping`
- `--output-dir`: (Optional) Output folder for cropped videos (default: `CroppedVideos` in parent of input_dir)
- `--threads`: Number of parallel threads (default: 1)
- `--encoder`: (Optional) Video encoder: `cpu` (libx264, default), `nvidia` (h264_nvenc), or `apple` (h264_videotoolbox)
- `--log-level`: (Optional) Logging verbosity (default: WARNING)

## Example

```sh
python wells_cropper.py /path/to/raw_videos /path/to/params.json --threads 8 --encoder apple
```

## Requirements
- [Tierpsy Tracker](https://github.com/Tierpsy/tierpsy-tracker) (with dependencies)
- OpenCV, FFmpeg, tqdm, numpy, pyyaml

## Notes

- For Apple Silicon, use `--encoder apple`
- For NVIDIA GPUs, use `--encoder nvidia`
- For maximum compatibility, use `--encoder cpu`
- The script is designed to work with videos with a specific directory structure and naming convention, 
where each videos are located in subfolders named as {plate_id + "." + Camera ID}. Well names are read based on "mwp_mapping" from the JSON file, 
which contains the well layout information source.
