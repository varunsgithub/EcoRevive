"""
Visualization utilities for California Fire Model
Generate prediction visualizations and comparisons.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from pathlib import Path
from typing import Optional, Tuple

# Add parent directory
sys.path.append(str(Path(__file__).parent.parent))

from config import SEVERITY_THRESHOLDS


# Burn severity colormap
SEVERITY_COLORS = [
    (0.0, '#2ECC71'),   # Unburned - Green
    (0.1, '#F1C40F'),   # Low - Yellow
    (0.27, '#E67E22'),  # Moderate-low - Orange
    (0.44, '#E74C3C'),  # Moderate-high - Red-orange
    (0.66, '#9B59B6'),  # High - Purple
    (1.0, '#1A1A2E'),   # Very high - Dark
]

def get_severity_cmap():
    """Get custom colormap for burn severity."""
    colors = [c[1] for c in SEVERITY_COLORS]
    positions = [c[0] for c in SEVERITY_COLORS]
    
    cmap = mcolors.LinearSegmentedColormap.from_list(
        'burn_severity',
        list(zip(positions, colors))
    )
    return cmap


def rgb_from_sentinel2(image: np.ndarray) -> np.ndarray:
    """
    Create RGB composite from Sentinel-2 bands.
    
    Args:
        image: (C, H, W) Sentinel-2 image with at least 4 bands
               Bands: B2, B3, B4, ... (Blue, Green, Red, ...)
               
    Returns:
        (H, W, 3) RGB image for visualization
    """
    # Bands 2, 3, 4 are Blue, Green, Red
    # For display: RGB = [Red, Green, Blue] = [B4, B3, B2]
    red = image[2]    # B4
    green = image[1]  # B3
    blue = image[0]   # B2
    
    rgb = np.stack([red, green, blue], axis=-1)
    
    # Normalize for display
    rgb = np.clip(rgb, 0, 3000)  # Typical surface reflectance range
    rgb = rgb / 3000
    
    return rgb


def plot_prediction(
    image: np.ndarray,
    prediction: np.ndarray,
    ground_truth: Optional[np.ndarray] = None,
    title: str = "Burn Severity Prediction",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 5),
) -> plt.Figure:
    """
    Visualize model prediction with RGB image and severity map.
    
    Args:
        image: (C, H, W) Sentinel-2 image
        prediction: (H, W) predicted severity [0, 1]
        ground_truth: (H, W) optional ground truth severity
        title: Figure title
        save_path: Optional path to save figure
        figsize: Figure size
        
    Returns:
        matplotlib Figure
    """
    n_cols = 3 if ground_truth is not None else 2
    fig, axes = plt.subplots(1, n_cols, figsize=figsize)
    
    cmap = get_severity_cmap()
    
    # RGB composite
    rgb = rgb_from_sentinel2(image)
    axes[0].imshow(rgb)
    axes[0].set_title("Sentinel-2 RGB")
    axes[0].axis('off')
    
    # Prediction
    im = axes[1].imshow(prediction, cmap=cmap, vmin=0, vmax=1)
    axes[1].set_title(f"Predicted Severity\n(Mean: {prediction.mean():.1%})")
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Ground truth (if available)
    if ground_truth is not None:
        im = axes[2].imshow(ground_truth, cmap=cmap, vmin=0, vmax=1)
        axes[2].set_title(f"Ground Truth\n(Mean: {ground_truth.mean():.1%})")
        axes[2].axis('off')
        plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"💾 Saved: {save_path}")
    
    return fig


def plot_comparison(
    predictions: list,
    labels: list,
    title: str = "Model Comparison",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 4),
) -> plt.Figure:
    """
    Compare multiple predictions side by side.
    
    Args:
        predictions: List of (H, W) prediction arrays
        labels: List of labels for each prediction
        title: Figure title
        save_path: Optional path to save figure
        
    Returns:
        matplotlib Figure
    """
    n = len(predictions)
    fig, axes = plt.subplots(1, n, figsize=figsize)
    
    if n == 1:
        axes = [axes]
    
    cmap = get_severity_cmap()
    
    for ax, pred, label in zip(axes, predictions, labels):
        im = ax.imshow(pred, cmap=cmap, vmin=0, vmax=1)
        ax.set_title(f"{label}\n(Mean: {pred.mean():.1%})")
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_temporal_series(
    predictions: dict,  # {'pre_fire': ..., 'post_fire': ..., 'recovery_1yr': ...}
    title: str = "Fire Progression",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Visualize temporal progression of burn severity.
    
    Args:
        predictions: Dict mapping stage name to (H, W) severity map
        title: Figure title
        save_path: Optional path to save figure
        
    Returns:
        matplotlib Figure
    """
    stages = list(predictions.keys())
    n = len(stages)
    
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]
    
    cmap = get_severity_cmap()
    
    for ax, stage in zip(axes, stages):
        pred = predictions[stage]
        im = ax.imshow(pred, cmap=cmap, vmin=0, vmax=1)
        
        # Clean up stage name for display
        display_name = stage.replace('_', ' ').title()
        ax.set_title(f"{display_name}\n({pred.mean():.1%} severity)")
        ax.axis('off')
    
    # Single colorbar
    fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.02, pad=0.04,
                 label='Burn Severity')
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_error_map(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    title: str = "Prediction Error",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Visualize prediction errors (difference from ground truth).
    
    Args:
        prediction: (H, W) predicted severity
        ground_truth: (H, W) ground truth severity
        title: Figure title
        save_path: Optional path to save figure
        
    Returns:
        matplotlib Figure
    """
    error = prediction - ground_truth  # Positive = overestimate
    abs_error = np.abs(error)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Signed error
    im0 = axes[0].imshow(error, cmap='RdBu_r', vmin=-1, vmax=1)
    axes[0].set_title(f"Signed Error\n(+: overestimate, -: underestimate)")
    axes[0].axis('off')
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    
    # Absolute error
    im1 = axes[1].imshow(abs_error, cmap='hot', vmin=0, vmax=1)
    axes[1].set_title(f"Absolute Error\n(MAE: {abs_error.mean():.4f})")
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Error histogram
    axes[2].hist(error.flatten(), bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    axes[2].axvline(0, color='red', linestyle='--', linewidth=2)
    axes[2].axvline(error.mean(), color='orange', linestyle='-', linewidth=2)
    axes[2].set_xlabel('Error')
    axes[2].set_ylabel('Pixel Count')
    axes[2].set_title(f"Error Distribution\n(Bias: {error.mean():+.4f})")
    axes[2].legend(['Zero', f'Mean = {error.mean():+.3f}'])
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_severity_legend(
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 1),
) -> plt.Figure:
    """Create a severity legend."""
    fig, ax = plt.subplots(figsize=figsize)
    
    cmap = get_severity_cmap()
    norm = mcolors.Normalize(vmin=0, vmax=1)
    
    cb = plt.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=ax,
        orientation='horizontal',
    )
    
    # Add severity labels
    cb.set_ticks([0.05, 0.18, 0.35, 0.55, 0.83])
    cb.set_ticklabels(['Unburned', 'Low', 'Mod-Low', 'Mod-High', 'High'])
    
    ax.set_title('Burn Severity Scale', fontsize=12)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


# ============================================================
# TEST
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("🎨 Testing Visualization")
    print("=" * 60)
    
    # Create dummy data
    np.random.seed(42)
    image = np.random.randint(0, 3000, (10, 256, 256)).astype(np.float32)
    prediction = np.clip(np.random.randn(256, 256) * 0.3 + 0.4, 0, 1)
    ground_truth = np.clip(np.random.randn(256, 256) * 0.3 + 0.5, 0, 1)
    
    # Test plotting
    print("\n📊 Creating visualizations...")
    
    fig1 = plot_prediction(image, prediction, ground_truth, title="Test Prediction")
    plt.close(fig1)
    print("   ✅ plot_prediction")
    
    fig2 = plot_comparison([prediction, ground_truth], ['Prediction', 'Ground Truth'])
    plt.close(fig2)
    print("   ✅ plot_comparison")
    
    fig3 = plot_error_map(prediction, ground_truth)
    plt.close(fig3)
    print("   ✅ plot_error_map")
    
    fig4 = plot_temporal_series({
        'pre_fire': np.zeros((256, 256)) + 0.1,
        'post_fire': prediction,
        'recovery_1yr': np.clip(prediction * 0.7, 0, 1),
    })
    plt.close(fig4)
    print("   ✅ plot_temporal_series")
    
    fig5 = plot_severity_legend()
    plt.close(fig5)
    print("   ✅ plot_severity_legend")
    
    print("\n✅ All visualizations working!")
    print("=" * 60)
