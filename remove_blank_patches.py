import argparse
import os
import warnings
import imageio
import numpy as np
from tqdm import tqdm
import multiprocessing

# Suppress warnings
warnings.filterwarnings("ignore")

def check_blank_worker(args):
    """Worker function to check a chunk of images for blankness."""
    files, blank_threshold, white_pixel_intensity_threshold = args
    blank_files = []
    errors = []
    for file_path in files:
        is_blank, err = is_nearly_blank(file_path, blank_threshold, white_pixel_intensity_threshold)
        if err:
            errors.append((os.path.basename(file_path), err))
        if is_blank:
            blank_files.append(file_path)
    return blank_files, errors

def is_nearly_blank(image_path, blank_threshold=95, white_pixel_intensity_threshold=240):
    try:
        img = imageio.imread(image_path)
    except Exception as e:
        return False, str(e)

    if img.ndim == 3:
        img = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
    
    white_pixels_count = np.sum(img > white_pixel_intensity_threshold)
    total_pixels = img.size
    
    percentage_white = (white_pixels_count / total_pixels) * 100
    return percentage_white >= blank_threshold, None

def remove_blank_patches_parallel(input_dir, blank_threshold, white_pixel_intensity_threshold, dry_run, num_workers):
    if not os.path.isdir(input_dir):
        print(f"Error: Input directory not found at {input_dir}")
        return

    all_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
    
    if not all_files:
        print("No image files found in the directory.")
        return

    file_chunks = np.array_split(all_files, num_workers)
    worker_args = [(chunk, blank_threshold, white_pixel_intensity_threshold) for chunk in file_chunks]

    blank_files_to_remove = []
    all_errors = []
    
    with multiprocessing.Pool(processes=num_workers) as pool:
        with tqdm(total=len(all_files), desc="Analyzing images") as pbar:
            for blank_files_chunk, errors_chunk in pool.imap_unordered(check_blank_worker, worker_args):
                blank_files_to_remove.extend(blank_files_chunk)
                all_errors.extend(errors_chunk)
                pbar.update(len(blank_files_chunk) + len(errors_chunk))

    if dry_run:
        print(f"\nFound {len(blank_files_to_remove)} blank images to remove (DRY RUN).")
        # for f in blank_files_to_remove:
        #     print(f"- {os.path.basename(f)}")
    else:
        if blank_files_to_remove:
            for f in tqdm(blank_files_to_remove, desc="Deleting blank images"):
                try:
                    os.remove(f)
                except Exception as e:
                    all_errors.append((os.path.basename(f), f"Error removing file: {e}"))
            print(f"\nRemoved {len(blank_files_to_remove)} blank images.")

    if all_errors:
        print("\nThe following errors occurred:")
        for filename, error_msg in all_errors:
            print(f"- {filename}: {error_msg}")

def main():
    parser = argparse.ArgumentParser(description="Remove nearly blank image patches from a directory in parallel.")
    parser.add_argument("-i", "--input-dir", required=True, help="Path to the directory with patches.")
    parser.add_argument("-t", "--threshold", type=int, default=95, help="Blankness threshold percentage (0-100).")
    parser.add_argument("-w", "--white-pixel-intensity", type=int, default=240, help="Intensity for 'white' pixels (0-255).")
    parser.add_argument("--dry-run", action="store_true", help="Perform a dry run without deleting files.")
    parser.add_argument("-nw", "--num-workers", type=int, default=multiprocessing.cpu_count(), help="Number of parallel workers.")
    
    args = parser.parse_args()

    remove_blank_patches_parallel(args.input_dir, args.threshold, args.white_pixel_intensity, args.dry_run, args.num_workers)

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()