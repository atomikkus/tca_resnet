import argparse
import os
import multiprocessing
import numpy as np
import imageio
from tqdm import tqdm
from tiatoolbox.tools.patchextraction import SlidingWindowPatchExtractor
from tiatoolbox.wsicore import WSIReader
import warnings

# Suppress warnings from tiatoolbox and underlying libraries
warnings.filterwarnings("ignore")

def save_patch_worker(args):
    """Worker function to save a chunk of patches."""
    wsi_path, locations, patch_size, output_dir, level, start_idx = args
    wsi = WSIReader.open(wsi_path)
    count = 0
    for i, row in enumerate(locations.itertuples()):
        patch = wsi.read_rect((row.x, row.y), (patch_size, patch_size), resolution=level, units="level")
        patch_path = os.path.join(output_dir, f"patch_{start_idx + i:06d}.png")
        imageio.imwrite(patch_path, patch)
        count += 1
    return count

def extract_patches_parallel(wsi_path, output_dir, patch_size, stride, level, num_workers):
    """
    Extracts patches from a WSI file in parallel and saves them as PNG images.
    """
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(wsi_path):
        print(f"Error: Input file not found at {wsi_path}")
        return

    try:
        wsi = WSIReader.open(wsi_path)
    except Exception as e:
        print(f"Error opening WSI file: {e}")
        return

    patch_extractor = SlidingWindowPatchExtractor(
        input_img=wsi,
        patch_size=(patch_size, patch_size),
        stride=(stride, stride),
        resolution=level,
        units="level",
    )

    locations_df = patch_extractor.locations_df
    num_patches = len(locations_df)
    if num_patches == 0:
        print("No patches to extract.")
        return
        
    location_chunks = np.array_split(locations_df, num_workers)
    
    # Correctly get start_indices for each chunk
    start_indices = [chunk.index[0] if not chunk.empty else 0 for chunk in location_chunks]

    worker_args = [
        (wsi_path, location_chunks[i], patch_size, output_dir, level, start_indices[i])
        for i in range(len(location_chunks))
    ]

    total_processed = 0
    with multiprocessing.Pool(processes=num_workers) as pool:
        with tqdm(total=num_patches, desc="Extracting patches") as pbar:
            for count in pool.imap_unordered(save_patch_worker, worker_args):
                pbar.update(count)
                total_processed += count

    print(f"\nSuccessfully extracted and saved {total_processed} patches to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Extract patches from a WSI file.")
    parser.add_argument("-i", "--input-file", required=True, help="Path to the WSI file.")
    parser.add_argument("-o", "--output-dir", required=True, help="Directory to save the patches.")
    parser.add_argument("-ps", "--patch-size", type=int, default=256, help="The size of the patches to extract.")
    parser.add_argument("-s", "--stride", type=int, default=256, help="The stride for the sliding window.")
    parser.add_argument("-l", "--level", type=int, default=0, help="The resolution level to extract from.")
    parser.add_argument("-nw", "--num-workers", type=int, default=multiprocessing.cpu_count(), help="Number of parallel workers.")

    args = parser.parse_args()

    extract_patches_parallel(
        args.input_file,
        args.output_dir,
        args.patch_size,
        args.stride,
        args.level,
        args.num_workers
    )

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
