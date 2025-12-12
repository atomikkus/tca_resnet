"""
Complete WSI Analysis Pipeline

This script provides a full pipeline for:
1. Loading a Whole Slide Image (WSI)
2. Extracting patches using Otsu thresholding (tissue mask)
3. Classifying patches into 9 tissue classes
4. Annotating predictions on WSI overview image

Classes:
- BACK: Background (empty glass region)
- NORM: Normal colon mucosa
- DEB: Debris
- TUM: Colorectal adenocarcinoma epithelium
- ADI: Adipose
- MUC: Mucus
- MUS: Smooth muscle
- STR: Cancer-associated stroma
- LYM: Lymphocytes
"""

import argparse
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd

from tiatoolbox import logger
from tiatoolbox.models.engine.patch_predictor import (
    IOPatchPredictorConfig,
    PatchPredictor,
)
from tiatoolbox.utils.visualization import overlay_prediction_mask
from tiatoolbox.wsicore.wsireader import WSIReader

# Configure matplotlib
plt.rcParams["figure.dpi"] = 150
plt.rcParams["figure.facecolor"] = "white"

# Class definitions
CLASS_NAMES = ["BACK", "NORM", "DEB", "TUM", "ADI", "MUC", "MUS", "STR", "LYM"]
NUM_CLASSES = len(CLASS_NAMES)

# Create label-color mapping for visualization
def create_label_color_dict(only_tum=False):
    """Create a label-color dictionary for visualization.
    
    Args:
        only_tum: If True, only include TUM class in the legend
    """
    label_color_dict = {}
    label_color_dict[0] = ("empty", (0, 0, 0))  # Background/empty regions
    
    if only_tum:
        # Only include TUM class in legend with blue color
        tum_index = CLASS_NAMES.index("TUM")
        label_color_dict[tum_index + 1] = ("TUM", (0, 0, 255))
    else:
        # Use Set1 colormap for all 9 classes
        try:
            # For matplotlib >= 3.7
            colors = plt.colormaps["Set1"].colors
        except AttributeError:
            # Fallback for older matplotlib versions
            colors = cm.get_cmap("Set1").colors
        for i, class_name in enumerate(CLASS_NAMES):
            label_color_dict[i + 1] = (class_name, tuple(int(255 * c) for c in colors[i]))
    
    return label_color_dict


def extract_patches_with_otsu(wsi_path, patch_size, stride, min_mask_ratio=0.1):
    """
    Extract patches from WSI using Otsu thresholding for tissue mask.
    
    Args:
        wsi_path: Path to WSI file
        patch_size: Tuple of (height, width) for patch size
        stride: Tuple of (height, width) for stride
        min_mask_ratio: Minimum ratio of patch that must overlap with tissue mask
    
    Returns:
        extractor: SlidingWindowPatchExtractor instance
        wsi: WSIReader instance
    """
    from tiatoolbox.tools.patchextraction import SlidingWindowPatchExtractor
    
    logger.info(f"Loading WSI: {wsi_path}")
    wsi = WSIReader.open(wsi_path)
    
    logger.info("Initializing patch extractor with Otsu thresholding...")
    extractor = SlidingWindowPatchExtractor(
        input_img=wsi_path,
        patch_size=patch_size,
        stride=stride,
        input_mask="otsu",
        min_mask_ratio=min_mask_ratio,
    )
    
    total_patches = len(extractor)
    logger.info(f"Found {total_patches} patches within tissue mask")
    
    return extractor, wsi


def predict_patches_with_otsu_extraction(
    wsi_path,
    model_name="resnet18-kather100k",
    patch_size=(224, 224),
    stride=(224, 224),
    batch_size=32,
    device="cpu",
    min_mask_ratio=0.1,
):
    """
    Alternative approach: Extract patches with Otsu mask first, then predict.
    This ensures proper Otsu masking support.
    
    Args:
        wsi_path: Path to WSI file
        model_name: Pretrained model name
        patch_size: Tuple of (height, width) for patch size
        stride: Tuple of (height, width) for stride
        batch_size: Batch size for prediction
        device: Device to use ('cuda' or 'cpu')
        min_mask_ratio: Minimum ratio of patch that must overlap with tissue mask
    
    Returns:
        predictor: PatchPredictor instance
        predictions: Array of predictions
        probabilities: Array of probabilities
        coordinates: Array of patch coordinates
    """
    from tiatoolbox.tools.patchextraction import SlidingWindowPatchExtractor
    
    logger.info("Extracting patches with Otsu mask...")
    
    # Extract patches using Otsu mask
    extractor = SlidingWindowPatchExtractor(
        input_img=wsi_path,
        patch_size=patch_size,
        stride=stride,
        input_mask="otsu",
        min_mask_ratio=min_mask_ratio,
    )
    
    total_patches = len(extractor)
    logger.info(f"Extracted {total_patches} patches within Otsu tissue mask")
    
    if total_patches == 0:
        logger.warning("No patches found within tissue mask")
        return None, None, None, None
    
    # Collect patches and coordinates
    patches = []
    coordinates = []
    
    logger.info("Loading patches into memory...")
    for i, patch in enumerate(extractor):
        patches.append(patch)
        coords = extractor.locations_df.iloc[i]
        coordinates.append([coords['x'], coords['y'], 
                           coords['x'] + patch_size[1], 
                           coords['y'] + patch_size[0]])
    
    patches = np.array(patches)
    coordinates = np.array(coordinates)
    
    logger.info(f"Patches shape: {patches.shape}")
    
    # Predict patches
    logger.info(f"Initializing PatchPredictor with model: {model_name}")
    predictor = PatchPredictor(
        pretrained_model=model_name,
        batch_size=batch_size
    )
    
    logger.info("Running predictions...")
    output = predictor.predict(
        imgs=patches,
        mode="patch",
        return_probabilities=True,
        device=device
    )
    
    predictions = output["predictions"]
    probabilities = output["probabilities"]
    
    logger.info("Prediction complete")
    
    return predictor, predictions, probabilities, coordinates


def predict_wsi_patches(
    wsi_path,
    model_name="resnet18-kather100k",
    patch_size=(224, 224),
    stride=(224, 224),
    resolution=0.5,
    resolution_units="mpp",
    batch_size=32,
    device="cpu",
    min_mask_ratio=0.1,
    merge_predictions=False,
    save_dir=None,
    use_otsu_mask=True,
):
    """
    Predict patches directly from WSI using PatchPredictor.
    
    Args:
        wsi_path: Path to WSI file
        model_name: Pretrained model name
        patch_size: Tuple of (height, width) for patch size
        stride: Tuple of (height, width) for stride
        resolution: Resolution for patch extraction
        resolution_units: Units for resolution ("mpp", "power", "level", "baseline")
        batch_size: Batch size for prediction
        device: Device to use ('cuda' or 'cpu')
        min_mask_ratio: Minimum ratio of patch that must overlap with tissue mask
        merge_predictions: Whether to merge predictions in memory
        save_dir: Directory to save intermediate results
        use_otsu_mask: Whether to use Otsu threshold mask
    
    Returns:
        predictor: PatchPredictor instance
        wsi_output: Prediction output dictionary
        mask_path: Path to mask file used (if any)
    """
    logger.info("Running predictions on WSI patches...")
    logger.info(f"  Patch size: {patch_size}")
    logger.info(f"  Stride: {stride}")
    logger.info(f"  Resolution: {resolution} {resolution_units}")
    
    # Configure IO for direct WSI prediction when not using Otsu mask
    ioconfig = IOPatchPredictorConfig(
        input_resolutions=[{"units": resolution_units, "resolution": resolution}],
        patch_input_shape=list(patch_size),
        stride_shape=list(stride),
    )

    # For Otsu masking, use patch extraction approach for better support
    if use_otsu_mask:
        logger.info("Using Otsu mask via patch extraction approach...")
        predictor, predictions, probabilities, coordinates = predict_patches_with_otsu_extraction(
            wsi_path=wsi_path,
            model_name=model_name,
            patch_size=patch_size,
            stride=stride,
            batch_size=batch_size,
            device=device,
            min_mask_ratio=min_mask_ratio,
        )
        
        if predictor is None:
            raise ValueError("No patches found within tissue mask")
        
        # Format output to match WSI mode output structure
        # Note: coordinates from SlidingWindowPatchExtractor are at baseline resolution
        wsi_output = [{
            "predictions": predictions.tolist() if isinstance(predictions, np.ndarray) else predictions,
            "probabilities": probabilities.tolist() if isinstance(probabilities, np.ndarray) else probabilities,
            "coordinates": coordinates.tolist() if isinstance(coordinates, np.ndarray) else coordinates,
            "resolution": 1.0,  # Baseline resolution
            "units": "baseline",
        }]
        mask_path = "otsu"  # Indicate Otsu was used
    else:
        logger.info("Processing all patches (no mask)...")
        wsi_output = predictor.predict(
            imgs=[wsi_path],
            masks=None,
            mode="wsi",
            merge_predictions=merge_predictions,
            ioconfig=ioconfig,
            return_probabilities=True,
            save_dir=save_dir,
            device=device,
        )
        mask_path = None
    
    logger.info("WSI prediction complete")
    
    return predictor, wsi_output, mask_path


def create_prediction_map(
    predictor,
    wsi_path,
    wsi_output,
    overview_resolution=4.0,
    overview_units="mpp",
    patch_size=(224, 224),
):
    """
    Create a prediction map from WSI output.
    
    Args:
        predictor: PatchPredictor instance
        wsi_path: Path to WSI file
        wsi_output: Prediction output from predictor
        overview_resolution: Resolution for overview image
        overview_units: Units for overview resolution
        patch_size: Patch size used (for manual mapping if needed)
    
    Returns:
        pred_map: Prediction map array
    """
    logger.info(f"Creating prediction map at {overview_resolution} {overview_units}")
    
    # Ensure output has resolution and units info
    if isinstance(wsi_output, list):
        output_dict = wsi_output[0]
    else:
        output_dict = wsi_output
    
    # Get WSI info for proper scaling
    wsi = WSIReader.open(wsi_path)
    wsi_info = wsi.info
    
    # If coordinates are in patch extraction format, try merge_predictions
    # Otherwise, create map manually
    if "coordinates" in output_dict and len(output_dict["coordinates"]) > 0:
        # Check if coordinates are in [x_min, y_min, x_max, y_max] format
        coords = output_dict["coordinates"][0]
        if len(coords) == 4:
            # Ensure output dict has proper format for merge_predictions
            output_dict_copy = output_dict.copy()
            if "resolution" not in output_dict_copy:
                # Get resolution from extraction resolution
                extraction_res = output_dict_copy.get("resolution", 0.5)
                extraction_units = output_dict_copy.get("units", "mpp")
                output_dict_copy["resolution"] = extraction_res
                output_dict_copy["units"] = extraction_units
            
            try:
                pred_map = predictor.merge_predictions(
                    wsi_path,
                    output_dict_copy,
                    resolution=overview_resolution,
                    units=overview_units,
                )
                logger.info(f"Prediction map shape: {pred_map.shape}")
                return pred_map
            except Exception as e:
                logger.warning(f"merge_predictions failed: {e}. Creating map manually...")
    
    # Manual mapping approach with proper coordinate scaling
    logger.info("Creating prediction map manually from coordinates...")
    wsi_overview = wsi.slide_thumbnail(
        resolution=overview_resolution,
        units=overview_units
    )
    
    h, w = wsi_overview.shape[:2]
    pred_map = np.zeros((h, w), dtype=np.int32)

    # Create tissue mask at overview resolution to filter predictions
    tissue_mask = None
    try:
        logger.info("Creating tissue mask at overview resolution...")
        from tiatoolbox.tools.tissuemask import OtsuTissueMasker

        masker = OtsuTissueMasker()
        tissue_mask = masker.fit_transform(wsi_overview)
        logger.info(
            "Tissue mask created (pixels inside mask: %d)",
            int(np.sum(tissue_mask > 0)),
        )
    except Exception as exc:
        logger.warning(
            "Could not create tissue mask at overview resolution: %s. Proceeding without additional masking.",
            exc,
        )

    # Scale baseline coordinates to overview thumbnail dimensions
    slide_w, slide_h = wsi_info.slide_dimensions
    scale_x = w / slide_w if slide_w else 1.0
    scale_y = h / slide_h if slide_h else 1.0
    logger.info(
        "Coordinate scaling factors -> scale_x: %.5f, scale_y: %.5f "
        "(slide dims: %d x %d, overview dims: %d x %d)",
        scale_x,
        scale_y,
        slide_w,
        slide_h,
        w,
        h,
    )

    predictions = output_dict["predictions"]
    coordinates = output_dict["coordinates"]
    
    # Convert to numpy arrays if needed
    if isinstance(predictions, list):
        predictions = np.array(predictions)
    if isinstance(coordinates, list):
        coordinates = np.array(coordinates)
    
    for coord, pred in zip(coordinates, predictions):
        x_min, y_min, x_max, y_max = coord
        
        # Scale coordinates based on thumbnail / baseline ratio
        x_min_scaled = int(x_min * scale_x)
        y_min_scaled = int(y_min * scale_y)
        x_max_scaled = int(x_max * scale_x)
        y_max_scaled = int(y_max * scale_y)
        
        # Ensure valid rectangle
        if x_max_scaled <= x_min_scaled:
            x_max_scaled = x_min_scaled + 1
        if y_max_scaled <= y_min_scaled:
            y_max_scaled = y_min_scaled + 1
        
        # Clip to image bounds
        x_min_scaled = max(0, min(x_min_scaled, w - 1))
        y_min_scaled = max(0, min(y_min_scaled, h - 1))
        x_max_scaled = max(x_min_scaled + 1, min(x_max_scaled, w))
        y_max_scaled = max(y_min_scaled + 1, min(y_max_scaled, h))
        
        # Only assign prediction if within tissue mask (if mask available)
        if tissue_mask is not None:
            patch_region = tissue_mask[
                y_min_scaled:y_max_scaled, x_min_scaled:x_max_scaled
            ]

            if patch_region.size == 0:
                continue

            mask_ratio = np.sum(patch_region > 0) / patch_region.size
            if mask_ratio < 0.3:
                continue

        # Assign prediction (add 1 because 0 is background)
        pred_map[y_min_scaled:y_max_scaled, x_min_scaled:x_max_scaled] = pred + 1
    
    logger.info(f"Prediction map shape: {pred_map.shape}")
    
    # Log statistics about prediction map
    unique_values, counts = np.unique(pred_map, return_counts=True)
    logger.info("Prediction map statistics:")
    for val, count in zip(unique_values, counts):
        if val == 0:
            logger.info(f"  Background (0): {count} pixels")
        elif val <= len(CLASS_NAMES):
            class_name = CLASS_NAMES[val - 1]
            logger.info(f"  {class_name} ({val}): {count} pixels")
    
    return pred_map


def visualize_wsi_annotations(
    wsi,
    pred_map,
    overview_resolution=4.0,
    overview_units="mpp",
    alpha=0.5,
    output_path=None,
):
    """
    Visualize predictions overlaid on WSI overview image.
    
    Args:
        wsi: WSIReader instance
        pred_map: Prediction map array
        overview_resolution: Resolution for overview image
        overview_units: Units for overview resolution
        alpha: Transparency for overlay (0-1)
        output_path: Path to save the visualization
    
    Returns:
        fig: matplotlib figure
    """
    logger.info(f"Generating WSI overview at {overview_resolution} {overview_units}")
    wsi_overview = wsi.slide_thumbnail(
        resolution=overview_resolution,
        units=overview_units
    )
    
    logger.info(f"WSI overview shape: {wsi_overview.shape}")
    
    # Filter pred_map to only show TUM class (label 4 since TUM is at index 3 and we add 1)
    # TUM is at index 3 in CLASS_NAMES
    tum_label = CLASS_NAMES.index("TUM") + 1
    tum_mask = (pred_map == tum_label)
    
    logger.info(f"Filtering to show only TUM class (label {tum_label})")
    tum_pixels = np.sum(tum_mask)
    logger.info(f"TUM pixels in annotation: {tum_pixels}")
    
    # Create manual overlay - only blue for TUM regions
    logger.info("Creating custom TUM overlay...")
    overlay_image = wsi_overview.copy()
    
    # Create blue overlay only where TUM is detected
    blue_color = np.array([0, 0, 255], dtype=np.uint8)
    
    # Apply blue overlay with alpha blending only to TUM regions
    for i in range(3):  # RGB channels
        overlay_image[:, :, i] = np.where(
            tum_mask,
            (1 - alpha) * wsi_overview[:, :, i] + alpha * blue_color[i],
            wsi_overview[:, :, i]
        ).astype(np.uint8)
    
    # Create figure without legend
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(overlay_image)
    ax.set_title("WSI with TUM Annotations", fontsize=14, fontweight='bold')
    ax.axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Visualization saved to {output_path}")
    
    return fig


def generate_summary_report(wsi_output, pred_map=None, output_dir=None):
    """
    Generate summary statistics from WSI predictions.
    
    Args:
        wsi_output: Prediction output from predictor
        pred_map: Prediction map array for area calculations (optional)
        output_dir: Directory to save report
    
    Returns:
        report: Dictionary with summary statistics
    """
    if isinstance(wsi_output, list):
        output_dict = wsi_output[0]
    else:
        output_dict = wsi_output
    
    predictions = output_dict.get("predictions", [])
    
    if len(predictions) == 0:
        logger.warning("No predictions found in output")
        return {}
    
    # Count predictions per class
    unique, counts = np.unique(predictions, return_counts=True)
    
    report = {
        "total_patches": len(predictions),
        "class_distribution": {}
    }
    
    for class_id, count in zip(unique, counts):
        class_name = CLASS_NAMES[class_id]
        percentage = (count / len(predictions)) * 100
        report["class_distribution"][class_name] = {
            "count": int(count),
            "percentage": round(percentage, 2)
        }
    
    # Fill in zeros for classes not present
    for i, class_name in enumerate(CLASS_NAMES):
        if class_name not in report["class_distribution"]:
            report["class_distribution"][class_name] = {
                "count": 0,
                "percentage": 0.0
            }
    
    # Calculate area coverage if pred_map is provided
    if pred_map is not None:
        logger.info("Calculating area coverage from prediction map...")
        
        # Get total tissue area (non-zero pixels)
        total_tissue_pixels = np.sum(pred_map > 0)
        
        # Calculate TUM area
        tum_label = CLASS_NAMES.index("TUM") + 1
        tum_pixels = np.sum(pred_map == tum_label)
        
        # Calculate area coverage for all classes
        report["area_coverage"] = {}
        for i, class_name in enumerate(CLASS_NAMES):
            class_label = i + 1
            class_pixels = np.sum(pred_map == class_label)
            
            # Percentage of total tissue area
            area_percentage = (class_pixels / total_tissue_pixels * 100) if total_tissue_pixels > 0 else 0.0
            
            report["area_coverage"][class_name] = {
                "pixels": int(class_pixels),
                "area_percentage": round(area_percentage, 2)
            }
        
        report["total_tissue_pixels"] = int(total_tissue_pixels)
        
        # Calculate TUM vs non-TUM ratio
        non_tum_pixels = total_tissue_pixels - tum_pixels
        tum_percentage = (tum_pixels / total_tissue_pixels * 100) if total_tissue_pixels > 0 else 0.0
        
        report["tum_analysis"] = {
            "tum_pixels": int(tum_pixels),
            "non_tum_pixels": int(non_tum_pixels),
            "tum_percentage": round(tum_percentage, 2),
            "non_tum_percentage": round(100 - tum_percentage, 2)
        }
        
        logger.info(f"TUM area coverage: {tum_percentage:.2f}% of total tissue area")
    
    # Save text report
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = output_dir / f"wsi_summary_{timestamp}.txt"
        
        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("WSI PATCH CLASSIFICATION SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Total Patches: {report['total_patches']}\n\n")
            f.write("-" * 80 + "\n")
            f.write("CLASS DISTRIBUTION (by patch count)\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'Class':<10} {'Count':<10} {'Percentage':<10}\n")
            f.write("-" * 80 + "\n")
            for class_name in CLASS_NAMES:
                dist = report["class_distribution"][class_name]
                f.write(f"{class_name:<10} {dist['count']:<10} {dist['percentage']:<10.2f}%\n")
            
            # Add area coverage section if available
            if "area_coverage" in report:
                f.write("\n" + "=" * 80 + "\n")
                f.write("AREA COVERAGE ANALYSIS (by pixel area)\n")
                f.write("=" * 80 + "\n\n")
                f.write(f"Total Tissue Pixels: {report['total_tissue_pixels']:,}\n\n")
                f.write("-" * 80 + "\n")
                f.write(f"{'Class':<10} {'Pixels':<15} {'Area %':<10}\n")
                f.write("-" * 80 + "\n")
                for class_name in CLASS_NAMES:
                    area = report["area_coverage"][class_name]
                    f.write(f"{class_name:<10} {area['pixels']:<15,} {area['area_percentage']:<10.2f}%\n")
                
                # Add TUM analysis section
                if "tum_analysis" in report:
                    f.write("\n" + "=" * 80 + "\n")
                    f.write("TUM AREA ANALYSIS\n")
                    f.write("=" * 80 + "\n\n")
                    tum = report["tum_analysis"]
                    f.write(f"TUM Pixels:         {tum['tum_pixels']:>15,} ({tum['tum_percentage']:>6.2f}%)\n")
                    f.write(f"Non-TUM Pixels:     {tum['non_tum_pixels']:>15,} ({tum['non_tum_percentage']:>6.2f}%)\n")
                    f.write(f"Total Tissue:       {report['total_tissue_pixels']:>15,} (100.00%)\n")
        
        logger.info(f"Summary report saved to {report_path}")
    
    return report


def main():
    parser = argparse.ArgumentParser(
        description="Complete WSI analysis pipeline: load, extract patches with Otsu, classify, and annotate",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-i", "--input-wsi",
        required=True,
        type=Path,
        help="Path to input WSI file"
    )
    parser.add_argument(
        "-m", "--model",
        default="resnet18-kather100k",
        help="Pretrained model name"
    )
    parser.add_argument(
        "-ps", "--patch-size",
        type=int,
        nargs=2,
        default=[224, 224],
        metavar=("HEIGHT", "WIDTH"),
        help="Patch size (height, width)"
    )
    parser.add_argument(
        "-s", "--stride",
        type=int,
        nargs=2,
        default=[224, 224],
        metavar=("HEIGHT", "WIDTH"),
        help="Stride size (height, width)"
    )
    parser.add_argument(
        "-r", "--resolution",
        type=float,
        default=0.5,
        help="Resolution for patch extraction (in specified units)"
    )
    parser.add_argument(
        "-ru", "--resolution-units",
        default="mpp",
        choices=["mpp", "power", "level", "baseline"],
        help="Units for resolution"
    )
    parser.add_argument(
        "-or", "--overview-resolution",
        type=float,
        default=4.0,
        help="Resolution for overview visualization (in specified units)"
    )
    parser.add_argument(
        "-ou", "--overview-units",
        default="mpp",
        choices=["mpp", "power", "level", "baseline"],
        help="Units for overview resolution"
    )
    parser.add_argument(
        "-b", "--batch-size",
        type=int,
        default=32,
        help="Batch size for prediction"
    )
    parser.add_argument(
        "-d", "--device",
        default="cpu",
        choices=["cuda", "cpu"],
        help="Device to use for prediction"
    )
    parser.add_argument(
        "-mmr", "--min-mask-ratio",
        type=float,
        default=0.1,
        help="Minimum ratio of patch that must overlap with tissue mask"
    )
    parser.add_argument(
        "-a", "--alpha",
        type=float,
        default=0.5,
        help="Transparency for overlay (0-1)"
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        default="./wsi_output",
        help="Output directory for results"
    )
    parser.add_argument(
        "--save-intermediate",
        action="store_true",
        help="Save intermediate prediction results"
    )
    parser.add_argument(
        "--no-otsu-mask",
        action="store_true",
        help="Disable Otsu threshold masking (process all patches)"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    logger.info("=" * 80)
    logger.info("WSI ANALYSIS PIPELINE")
    logger.info("=" * 80)
    logger.info(f"Input WSI: {args.input_wsi}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Patch size: {args.patch_size}")
    logger.info(f"Stride: {args.stride}")
    logger.info(f"Device: {args.device}")
    
    # Step 1: Load WSI and verify
    logger.info("\n" + "=" * 80)
    logger.info("STEP 1: Loading WSI")
    logger.info("=" * 80)
    wsi = WSIReader.open(args.input_wsi)
    logger.info(f"WSI dimensions: {wsi.info.slide_dimensions}")
    logger.info(f"WSI levels: {wsi.info.level_count}")
    
    # Step 2: Extract patches and predict
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: Extracting patches and predicting")
    logger.info("=" * 80)
    
    save_dir = args.output_dir / "intermediate" if args.save_intermediate else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
    
    predictor, wsi_output, mask_path = predict_wsi_patches(
        wsi_path=args.input_wsi,
        model_name=args.model,
        patch_size=tuple(args.patch_size),
        stride=tuple(args.stride),
        resolution=args.resolution,
        resolution_units=args.resolution_units,
        batch_size=args.batch_size,
        device=args.device,
        min_mask_ratio=args.min_mask_ratio,
        merge_predictions=False,
        save_dir=save_dir,
        use_otsu_mask=not args.no_otsu_mask,
    )
    
    # Step 3: Create prediction map
    logger.info("\n" + "=" * 80)
    logger.info("STEP 3: Creating prediction map")
    logger.info("=" * 80)
    pred_map = create_prediction_map(
        predictor,
        args.input_wsi,
        wsi_output,
        overview_resolution=args.overview_resolution,
        overview_units=args.overview_units,
        patch_size=tuple(args.patch_size),
    )
    
    # Step 4: Generate summary report with area coverage
    logger.info("\n" + "=" * 80)
    logger.info("STEP 4: Generating summary report with area analysis")
    logger.info("=" * 80)
    report = generate_summary_report(wsi_output, pred_map=pred_map, output_dir=args.output_dir)
    
    # Print summary to console
    logger.info(f"\nTotal patches classified: {report['total_patches']}")
    logger.info("\nClass distribution (by patch count):")
    for class_name in CLASS_NAMES:
        dist = report["class_distribution"][class_name]
        logger.info(f"  {class_name}: {dist['count']} ({dist['percentage']:.2f}%)")
    
    if "tum_analysis" in report:
        logger.info("\nTUM Area Analysis:")
        tum = report["tum_analysis"]
        logger.info(f"  TUM coverage: {tum['tum_percentage']:.2f}% of tissue area")
        logger.info(f"  TUM pixels: {tum['tum_pixels']:,}")
        logger.info(f"  Non-TUM pixels: {tum['non_tum_pixels']:,}")
    
    # Step 5: Visualize annotations
    logger.info("\n" + "=" * 80)
    logger.info("STEP 5: Creating annotated visualization")
    logger.info("=" * 80)
    output_path = args.output_dir / f"wsi_annotated_{timestamp}.png"
    
    fig = visualize_wsi_annotations(
        wsi,
        pred_map,
        overview_resolution=args.overview_resolution,
        overview_units=args.overview_units,
        alpha=args.alpha,
        output_path=output_path,
    )
    
    # Save prediction map as numpy array
    pred_map_path = args.output_dir / f"prediction_map_{timestamp}.npy"
    np.save(pred_map_path, pred_map)
    logger.info(f"Prediction map saved to {pred_map_path}")
    
    # Save predictions to CSV if available
    if isinstance(wsi_output, list) and len(wsi_output) > 0:
        output_dict = wsi_output[0]
        if "predictions" in output_dict and "coordinates" in output_dict:
            csv_path = args.output_dir / f"predictions_{timestamp}.csv"
            df = pd.DataFrame({
                'x_min': [coord[0] for coord in output_dict["coordinates"]],
                'y_min': [coord[1] for coord in output_dict["coordinates"]],
                'x_max': [coord[2] for coord in output_dict["coordinates"]],
                'y_max': [coord[3] for coord in output_dict["coordinates"]],
                'predicted_class': [CLASS_NAMES[p] for p in output_dict["predictions"]],
                'predicted_label': output_dict["predictions"]
            })
            if "probabilities" in output_dict:
                probs = output_dict["probabilities"]
                # Convert to numpy array if it's a list
                if isinstance(probs, list):
                    probs = np.array(probs)
                for i, class_name in enumerate(CLASS_NAMES):
                    df[f'prob_{class_name}'] = probs[:, i]
            df.to_csv(csv_path, index=False)
            logger.info(f"Predictions CSV saved to {csv_path}")
    
    logger.info("\n" + "=" * 80)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Results saved to: {args.output_dir}")
    logger.info(f"  - Annotated WSI: {output_path}")
    logger.info(f"  - Prediction map: {pred_map_path}")
    
    # Show plot
    plt.show()


if __name__ == "__main__":
    main()

