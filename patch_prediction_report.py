"""
Patch Prediction Script with Comprehensive Report

This script predicts 9 tissue classes from image patches using TIAToolbox's
pretrained ResNet18 model trained on Kather 100k dataset.

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

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)

from tiatoolbox import logger
from tiatoolbox.models.engine.patch_predictor import PatchPredictor

# Configure matplotlib
plt.rcParams["figure.dpi"] = 150
plt.rcParams["figure.facecolor"] = "white"
sns.set_style("whitegrid")

# Class definitions
CLASS_NAMES = ["BACK", "NORM", "DEB", "TUM", "ADI", "MUC", "MUS", "STR", "LYM"]
NUM_CLASSES = len(CLASS_NAMES)


def load_patches_from_h5(h5_path: Path, max_patches: int = None):
    """
    Load patches from HDF5 file.
    
    Args:
        h5_path: Path to HDF5 file containing patches
        max_patches: Maximum number of patches to load (None for all)
    
    Returns:
        patches: numpy array of patches (N, H, W, 3)
        coords: numpy array of coordinates (N, 2) or None
    """
    logger.info(f"Loading patches from {h5_path}")
    
    with h5py.File(h5_path, 'r') as hf:
        patches = hf['patches'][:]
        coords = hf.get('coords', None)
        if coords is not None:
            coords = coords[:]
    
    if max_patches is not None and len(patches) > max_patches:
        logger.info(f"Limiting to {max_patches} patches")
        patches = patches[:max_patches]
        if coords is not None:
            coords = coords[:max_patches]
    
    logger.info(f"Loaded {len(patches)} patches")
    logger.info(f"Patch shape: {patches.shape}")
    
    return patches, coords


def predict_patches(patches, model_name="resnet18-kather100k", batch_size=32, device="cpu"):
    """
    Predict class labels for patches using TIAToolbox PatchPredictor.
    
    Args:
        patches: numpy array of patches (N, H, W, 3)
        model_name: Name of pretrained model
        batch_size: Batch size for prediction
        device: Device to use ('cuda' or 'cpu')
    
    Returns:
        predictions: Predicted class labels (N,)
        probabilities: Class probabilities (N, num_classes)
    """
    logger.info(f"Initializing PatchPredictor with model: {model_name}")
    predictor = PatchPredictor(
        pretrained_model=model_name,
        batch_size=batch_size
    )
    
    logger.info(f"Running predictions on {len(patches)} patches...")
    output = predictor.predict(
        imgs=patches,
        mode="patch",
        return_probabilities=True,
        device=device
    )
    
    predictions = output["predictions"]
    probabilities = output["probabilities"]
    
    logger.info("Prediction complete")
    
    return predictions, probabilities


def generate_report(predictions, probabilities=None, true_labels=None, output_dir=None):
    """
    Generate comprehensive classification report.
    
    Args:
        predictions: Predicted class labels (N,)
        probabilities: Class probabilities (N, num_classes) - optional
        true_labels: True class labels (N,) - optional, for evaluation
        output_dir: Directory to save report files
    """
    if output_dir is None:
        output_dir = Path("./reports")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create report dictionary
    report = {
        "timestamp": timestamp,
        "total_patches": len(predictions),
        "class_distribution": {},
        "metrics": {}
    }
    
    # Class distribution
    unique, counts = np.unique(predictions, return_counts=True)
    for class_id, count in zip(unique, counts):
        class_name = CLASS_NAMES[class_id]
        percentage = (count / len(predictions)) * 100
        report["class_distribution"][class_name] = {
            "count": int(count),
            "percentage": round(percentage, 2)
        }
    
    # If true labels provided, calculate metrics
    if true_labels is not None:
        acc = accuracy_score(true_labels, predictions)
        precision, recall, f1, support = precision_recall_fscore_support(
            true_labels, predictions, labels=range(NUM_CLASSES), zero_division=0
        )
        
        report["metrics"]["overall_accuracy"] = round(float(acc), 4)
        report["metrics"]["per_class"] = {}
        
        for i, class_name in enumerate(CLASS_NAMES):
            report["metrics"]["per_class"][class_name] = {
                "precision": round(float(precision[i]), 4),
                "recall": round(float(recall[i]), 4),
                "f1_score": round(float(f1[i]), 4),
                "support": int(support[i])
            }
    
    # Save text report
    report_path = output_dir / f"prediction_report_{timestamp}.txt"
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("PATCH PREDICTION REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Total Patches: {report['total_patches']}\n\n")
        
        f.write("-" * 80 + "\n")
        f.write("CLASS DISTRIBUTION\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Class':<10} {'Count':<10} {'Percentage':<10}\n")
        f.write("-" * 80 + "\n")
        for class_name in CLASS_NAMES:
            if class_name in report["class_distribution"]:
                dist = report["class_distribution"][class_name]
                f.write(f"{class_name:<10} {dist['count']:<10} {dist['percentage']:<10.2f}%\n")
            else:
                f.write(f"{class_name:<10} {0:<10} {0.00:<10.2f}%\n")
        
        if true_labels is not None:
            f.write("\n" + "-" * 80 + "\n")
            f.write("CLASSIFICATION METRICS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Overall Accuracy: {report['metrics']['overall_accuracy']:.4f}\n\n")
            f.write(f"{'Class':<10} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}\n")
            f.write("-" * 80 + "\n")
            for class_name in CLASS_NAMES:
                metrics = report["metrics"]["per_class"][class_name]
                f.write(f"{class_name:<10} {metrics['precision']:<12.4f} "
                       f"{metrics['recall']:<12.4f} {metrics['f1_score']:<12.4f} "
                       f"{metrics['support']:<10}\n")
            
            # Classification report
            f.write("\n" + "-" * 80 + "\n")
            f.write("DETAILED CLASSIFICATION REPORT\n")
            f.write("-" * 80 + "\n")
            f.write(classification_report(
                true_labels, predictions,
                target_names=CLASS_NAMES,
                labels=range(NUM_CLASSES)
            ))
    
    logger.info(f"Text report saved to {report_path}")
    
    # Create visualizations
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Class distribution bar plot
    ax1 = plt.subplot(2, 3, 1)
    class_counts = [report["class_distribution"].get(name, {}).get("count", 0) 
                   for name in CLASS_NAMES]
    bars = ax1.bar(CLASS_NAMES, class_counts, color=plt.cm.Set3(np.linspace(0, 1, NUM_CLASSES)))
    ax1.set_xlabel("Class", fontsize=10)
    ax1.set_ylabel("Number of Patches", fontsize=10)
    ax1.set_title("Class Distribution", fontsize=12, fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)
    for bar, count in zip(bars, class_counts):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}\n({count/len(predictions)*100:.1f}%)',
                ha='center', va='bottom', fontsize=8)
    
    # 2. Class distribution pie chart
    ax2 = plt.subplot(2, 3, 2)
    non_zero_counts = [(name, count) for name, count in zip(CLASS_NAMES, class_counts) if count > 0]
    if non_zero_counts:
        names, counts = zip(*non_zero_counts)
        ax2.pie(counts, labels=names, autopct='%1.1f%%', startangle=90)
        ax2.set_title("Class Distribution (Pie Chart)", fontsize=12, fontweight='bold')
    
    # 3. Confusion matrix (if true labels available)
    if true_labels is not None:
        ax3 = plt.subplot(2, 3, 3)
        cm = confusion_matrix(true_labels, predictions, labels=range(NUM_CLASSES))
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
                   xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                   ax=ax3, cbar_kws={'label': 'Normalized Count'})
        ax3.set_xlabel("Predicted", fontsize=10)
        ax3.set_ylabel("True", fontsize=10)
        ax3.set_title("Confusion Matrix (Normalized)", fontsize=12, fontweight='bold')
        ax3.tick_params(axis='x', rotation=45)
        ax3.tick_params(axis='y', rotation=0)
        
        # 4. Per-class metrics bar plot
        ax4 = plt.subplot(2, 3, 4)
        metrics_data = []
        for class_name in CLASS_NAMES:
            metrics_data.append({
                'Class': class_name,
                'Precision': report["metrics"]["per_class"][class_name]["precision"],
                'Recall': report["metrics"]["per_class"][class_name]["recall"],
                'F1-Score': report["metrics"]["per_class"][class_name]["f1_score"]
            })
        metrics_df = pd.DataFrame(metrics_data)
        x = np.arange(len(CLASS_NAMES))
        width = 0.25
        ax4.bar(x - width, metrics_df['Precision'], width, label='Precision', alpha=0.8)
        ax4.bar(x, metrics_df['Recall'], width, label='Recall', alpha=0.8)
        ax4.bar(x + width, metrics_df['F1-Score'], width, label='F1-Score', alpha=0.8)
        ax4.set_xlabel("Class", fontsize=10)
        ax4.set_ylabel("Score", fontsize=10)
        ax4.set_title("Per-Class Metrics", fontsize=12, fontweight='bold')
        ax4.set_xticks(x)
        ax4.set_xticklabels(CLASS_NAMES, rotation=45, ha='right')
        ax4.legend()
        ax4.set_ylim([0, 1.1])
        ax4.grid(axis='y', alpha=0.3)
    
    # 5. Probability distribution (if probabilities available)
    if probabilities is not None:
        ax5 = plt.subplot(2, 3, 5)
        mean_probs = np.mean(probabilities, axis=0)
        std_probs = np.std(probabilities, axis=0)
        x_pos = np.arange(len(CLASS_NAMES))
        ax5.bar(x_pos, mean_probs, yerr=std_probs, capsize=5, alpha=0.7,
               color=plt.cm.Set3(np.linspace(0, 1, NUM_CLASSES)))
        ax5.set_xlabel("Class", fontsize=10)
        ax5.set_ylabel("Mean Probability", fontsize=10)
        ax5.set_title("Mean Class Probabilities", fontsize=12, fontweight='bold')
        ax5.set_xticks(x_pos)
        ax5.set_xticklabels(CLASS_NAMES, rotation=45, ha='right')
        ax5.grid(axis='y', alpha=0.3)
    
    # 6. Summary statistics
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    summary_text = f"""
SUMMARY STATISTICS
{'='*50}

Total Patches: {report['total_patches']}
Number of Classes: {NUM_CLASSES}

Top 3 Predicted Classes:
"""
    sorted_classes = sorted(
        report["class_distribution"].items(),
        key=lambda x: x[1]["count"],
        reverse=True
    )[:3]
    for i, (class_name, dist) in enumerate(sorted_classes, 1):
        summary_text += f"{i}. {class_name}: {dist['count']} ({dist['percentage']:.2f}%)\n"
    
    if true_labels is not None:
        summary_text += f"\nOverall Accuracy: {report['metrics']['overall_accuracy']:.4f}\n"
        avg_f1 = np.mean([report["metrics"]["per_class"][name]["f1_score"] 
                         for name in CLASS_NAMES])
        summary_text += f"Average F1-Score: {avg_f1:.4f}\n"
    
    ax6.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
            verticalalignment='center', transform=ax6.transAxes)
    
    plt.tight_layout()
    
    # Save figure
    fig_path = output_dir / f"prediction_report_{timestamp}.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    logger.info(f"Visualization saved to {fig_path}")
    
    # Save CSV report
    csv_path = output_dir / f"predictions_{timestamp}.csv"
    df = pd.DataFrame({
        'patch_id': range(len(predictions)),
        'predicted_class': [CLASS_NAMES[p] for p in predictions],
        'predicted_label': predictions
    })
    if probabilities is not None:
        for i, class_name in enumerate(CLASS_NAMES):
            df[f'prob_{class_name}'] = probabilities[:, i]
    if true_labels is not None:
        df['true_class'] = [CLASS_NAMES[t] for t in true_labels]
        df['true_label'] = true_labels
        df['correct'] = (predictions == true_labels)
    df.to_csv(csv_path, index=False)
    logger.info(f"CSV report saved to {csv_path}")
    
    plt.close()
    
    return report


def main():
    parser = argparse.ArgumentParser(
        description="Predict 9 tissue classes from patches and generate comprehensive report",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-i", "--input",
        required=True,
        type=Path,
        help="Path to input HDF5 file containing patches"
    )
    parser.add_argument(
        "-m", "--model",
        default="resnet18-kather100k",
        help="Pretrained model name"
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
        "-o", "--output-dir",
        type=Path,
        default="./reports",
        help="Output directory for reports"
    )
    parser.add_argument(
        "--max-patches",
        type=int,
        default=None,
        help="Maximum number of patches to process (for testing)"
    )
    parser.add_argument(
        "--true-labels",
        type=Path,
        default=None,
        help="Path to file with true labels (for evaluation). "
             "Can be CSV with 'label' column or numpy array file"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Load patches
    patches, coords = load_patches_from_h5(args.input, args.max_patches)
    
    # Load true labels if provided
    true_labels = None
    if args.true_labels:
        logger.info(f"Loading true labels from {args.true_labels}")
        if args.true_labels.suffix == '.csv':
            df = pd.read_csv(args.true_labels)
            true_labels = df['label'].values
        else:
            true_labels = np.load(args.true_labels)
        logger.info(f"Loaded {len(true_labels)} true labels")
    
    # Predict patches
    predictions, probabilities = predict_patches(
        patches,
        model_name=args.model,
        batch_size=args.batch_size,
        device=args.device
    )
    
    # Generate report
    report = generate_report(
        predictions,
        probabilities=probabilities,
        true_labels=true_labels,
        output_dir=args.output_dir
    )
    
    logger.info("=" * 80)
    logger.info("PREDICTION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Total patches processed: {report['total_patches']}")
    if true_labels is not None:
        logger.info(f"Overall accuracy: {report['metrics']['overall_accuracy']:.4f}")
    logger.info(f"Reports saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

