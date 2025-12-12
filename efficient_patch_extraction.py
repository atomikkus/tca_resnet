import argparse
import os
import h5py
import numpy as np
from tqdm import tqdm
from tiatoolbox.wsicore.wsireader import WSIReader
from tiatoolbox.tools.patchextraction import SlidingWindowPatchExtractor

def is_nearly_blank_memory(patch_array, blank_threshold=95, white_pixel_intensity_threshold=240):
    """
    Checks if a patch (as a NumPy array) is nearly blank.
    """
    if patch_array.ndim == 3:
        # Convert to grayscale using ITU-R 601-2 luma transform
        img_gray = np.dot(patch_array[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
    else:
        img_gray = patch_array

    white_pixels_count = np.sum(img_gray > white_pixel_intensity_threshold)
    total_pixels = img_gray.size
    
    percentage_white = (white_pixels_count / total_pixels) * 100
    return percentage_white >= blank_threshold

def extract_and_filter_patches(
    wsi_path,
    output_h5_path,
    patch_size,
    stride,
    blank_threshold,
    white_pixel_intensity,
    min_mask_ratio
):
    """
    Extracts patches from a WSI, filters out blank ones in-memory,
    and saves the valid patches to an HDF5 file.
    """
    if not os.path.exists(wsi_path):
        print(f"Error: WSI file not found at {wsi_path}")
        return

    print("Step 1: Initializing patch extractor with Otsu's thresholding...")
    try:
        extractor = SlidingWindowPatchExtractor(
            input_img=wsi_path,
            patch_size=patch_size,
            stride=stride,
            input_mask="otsu",
            min_mask_ratio=min_mask_ratio,
        )
    except Exception as e:
        print(f"Error initializing patch extractor: {e}")
        print("This may be due to an issue with the WSI file or tiatoolbox setup.")
        return

    total_patches_in_mask = len(extractor)
    if total_patches_in_mask == 0:
        print("No patches found within the tissue mask. Exiting.")
        return

    print(f"Found {total_patches_in_mask} patches within the initial tissue mask.")
    print("Step 2: Iterating, filtering blank patches, and saving to HDF5...")

    h5_file = h5py.File(output_h5_path, 'w')
    # Create a resizable dataset for the patches
    dset = h5_file.create_dataset(
        'patches',
        (0, *patch_size, 3),
        maxshape=(None, *patch_size, 3),
        dtype=np.uint8,
        chunks=(1, *patch_size, 3) # Chunking for efficient reading later
    )
    
    # Create a dataset for coordinates
    coord_dset = h5_file.create_dataset(
        'coords',
        (0, 2),
        maxshape=(None, 2),
        dtype=np.int64,
        chunks=True
    )


    saved_count = 0
    with tqdm(total=total_patches_in_mask, desc="Processing patches") as pbar:
        for i, patch in enumerate(extractor):
            if not is_nearly_blank_memory(patch, blank_threshold, white_pixel_intensity):
                # Get coordinates
                coords = extractor.locations_df.iloc[i]
                
                # Expand dataset and save patch and coordinates
                dset.resize(saved_count + 1, axis=0)
                dset[saved_count] = patch
                
                coord_dset.resize(saved_count + 1, axis=0)
                coord_dset[saved_count] = [coords['x'], coords['y']]

                saved_count += 1
            pbar.update(1)

    h5_file.close()
    
    print("\n--- Process Complete ---")
    print(f"Total patches considered (from tissue mask): {total_patches_in_mask}")
    print(f"Total valid (non-blank) patches saved: {saved_count}")
    print(f"Output saved to: {output_h5_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Efficiently extract non-blank patches from a WSI and save to HDF5.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-i", "--input-wsi", required=True, help="Path to the input Whole Slide Image (SVS file).")
    parser.add_argument("-o", "--output-h5", required=True, help="Path to the output HDF5 file.")
    parser.add_argument("-ps", "--patch-size", type=int, default=512, help="Patch size (width and height).")
    parser.add_argument("-s", "--stride", type=int, default=512, help="Stride for patch extraction.")
    parser.add_argument("-bt", "--blank-threshold", type=int, default=95, help="Blankness threshold percentage (0-100).")
    parser.add_argument("-wpi", "--white-pixel-intensity", type=int, default=220, help="Intensity for 'white' pixels (0-255).")
    parser.add_argument("-mmr", "--min-mask-ratio", type=float, default=0.1, help="Minimum ratio of patch that must overlap with tissue mask (0-1).")

    args = parser.parse_args()

    patch_dims = (args.patch_size, args.patch_size)
    stride_dims = (args.stride, args.stride)

    extract_and_filter_patches(
        args.input_wsi,
        args.output_h5,
        patch_dims,
        stride_dims,
        args.blank_threshold,
        args.white_pixel_intensity,
        args.min_mask_ratio
    )

if __name__ == "__main__":
    main()