"""
Metrics for California Fire Model
Evaluation metrics for burn severity prediction.

Tracks both regression metrics (MAE, MSE) and segmentation metrics (IoU, Dice).
Importantly, tracks per-fire performance for debugging generalization.
"""

import torch
import numpy as np
from typing import Dict, List, Optional
from collections import defaultdict


class MetricTracker:
    """
    Tracks metrics across batches and provides summaries.
    Supports per-category (per-fire) breakdowns.
    """
    
    def __init__(self, threshold: float = 0.5):
        """
        Args:
            threshold: Threshold for binary metrics (e.g., IoU at 50%)
        """
        self.threshold = threshold
        self.reset()
    
    def reset(self):
        """Reset all tracked metrics."""
        self.metrics = defaultdict(list)
        self.metrics_by_category = defaultdict(lambda: defaultdict(list))
    
    def update(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        category: Optional[str] = None,
    ):
        """
        Update metrics with a batch of predictions.
        
        Args:
            logits: Model output (B, 1, H, W)
            targets: Ground truth (B, 1, H, W)
            category: Optional category (e.g., fire name) for per-category tracking
        """
        with torch.no_grad():
            probs = torch.sigmoid(logits)
            
            # Flatten for batch-wise metrics
            probs_flat = probs.view(probs.size(0), -1)
            targets_flat = targets.view(targets.size(0), -1)
            
            # Compute metrics per sample in batch
            for i in range(probs.size(0)):
                p = probs_flat[i]
                t = targets_flat[i]
                
                # Regression metrics
                mae = torch.abs(p - t).mean().item()
                mse = ((p - t) ** 2).mean().item()
                
                # Binarized metrics
                p_bin = (p > self.threshold).float()
                t_bin = (t > self.threshold).float()
                
                # IoU
                intersection = (p_bin * t_bin).sum().item()
                union = p_bin.sum().item() + t_bin.sum().item() - intersection
                iou = intersection / (union + 1e-6)
                
                # Dice
                dice = (2 * intersection) / (p_bin.sum().item() + t_bin.sum().item() + 1e-6)
                
                # Pixel accuracy
                correct = (p_bin == t_bin).float().mean().item()
                
                # Classification metrics (at threshold)
                tp = intersection
                fp = (p_bin * (1 - t_bin)).sum().item()
                fn = ((1 - p_bin) * t_bin).sum().item()
                
                precision = tp / (tp + fp + 1e-6)
                recall = tp / (tp + fn + 1e-6)
                f1 = 2 * precision * recall / (precision + recall + 1e-6)
                
                # Store metrics
                batch_metrics = {
                    'mae': mae,
                    'mse': mse,
                    'iou': iou,
                    'dice': dice,
                    'accuracy': correct,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'pred_mean': p.mean().item(),
                    'target_mean': t.mean().item(),
                }
                
                for key, value in batch_metrics.items():
                    self.metrics[key].append(value)
                    
                    if category is not None:
                        self.metrics_by_category[category][key].append(value)
    
    def get_summary(self) -> Dict[str, float]:
        """Get summary of all metrics."""
        summary = {}
        
        for key, values in self.metrics.items():
            if values:
                summary[key] = np.mean(values)
                summary[f'{key}_std'] = np.std(values)
        
        return summary
    
    def get_category_summary(self) -> Dict[str, Dict[str, float]]:
        """Get per-category metric summaries."""
        category_summary = {}
        
        for category, metrics in self.metrics_by_category.items():
            category_summary[category] = {}
            for key, values in metrics.items():
                if values:
                    category_summary[category][key] = np.mean(values)
        
        return category_summary
    
    def print_summary(self, prefix: str = ""):
        """Print formatted summary."""
        summary = self.get_summary()
        
        print(f"\n{prefix}Metrics Summary:")
        print("-" * 50)
        
        # Main metrics
        main_metrics = ['mae', 'mse', 'iou', 'dice', 'accuracy', 'f1']
        for key in main_metrics:
            if key in summary:
                print(f"   {key.upper():<12}: {summary[key]:.4f} ± {summary[f'{key}_std']:.4f}")
        
        # Per-category if available
        cat_summary = self.get_category_summary()
        if cat_summary:
            print(f"\n{prefix}Per-Category IoU:")
            for category, metrics in sorted(cat_summary.items()):
                if 'iou' in metrics:
                    print(f"   {category:<25}: {metrics['iou']:.4f}")


def compute_confusion_matrix(
    probs: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute confusion matrix metrics.
    
    Returns:
        Dict with TP, TN, FP, FN counts and rates
    """
    with torch.no_grad():
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        
        pred_pos = (probs_flat > threshold).float()
        pred_neg = 1 - pred_pos
        
        target_pos = (targets_flat > threshold).float()
        target_neg = 1 - target_pos
        
        tp = (pred_pos * target_pos).sum().item()
        tn = (pred_neg * target_neg).sum().item()
        fp = (pred_pos * target_neg).sum().item()
        fn = (pred_neg * target_pos).sum().item()
        
        total = tp + tn + fp + fn
        
        return {
            'tp': tp,
            'tn': tn,
            'fp': fp,
            'fn': fn,
            'tpr': tp / (tp + fn + 1e-6),  # Recall
            'tnr': tn / (tn + fp + 1e-6),  # Specificity
            'fpr': fp / (tn + fp + 1e-6),  # False positive rate
            'fnr': fn / (tp + fn + 1e-6),  # False negative rate
            'accuracy': (tp + tn) / total,
        }


def compute_severity_metrics(
    probs: torch.Tensor,
    targets: torch.Tensor,
    thresholds: Dict[str, tuple] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Compute metrics per burn severity category.
    
    Args:
        probs: Predicted probabilities (B, 1, H, W)
        targets: Ground truth (B, 1, H, W)
        thresholds: Dict mapping severity name to (min, max) ranges
        
    Returns:
        Dict with metrics per severity category
    """
    if thresholds is None:
        thresholds = {
            'unburned': (0.0, 0.1),
            'low': (0.1, 0.3),
            'moderate': (0.3, 0.6),
            'high': (0.6, 1.0),
        }
    
    with torch.no_grad():
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        
        results = {}
        
        for severity, (low, high) in thresholds.items():
            # Mask for this severity range in targets
            mask = (targets_flat >= low) & (targets_flat < high)
            
            if mask.sum() > 0:
                p_masked = probs_flat[mask]
                t_masked = targets_flat[mask]
                
                mae = torch.abs(p_masked - t_masked).mean().item()
                mse = ((p_masked - t_masked) ** 2).mean().item()
                bias = (p_masked - t_masked).mean().item()  # Positive = overestimate
                
                results[severity] = {
                    'count': mask.sum().item(),
                    'mae': mae,
                    'mse': mse,
                    'bias': bias,
                    'pred_mean': p_masked.mean().item(),
                    'target_mean': t_masked.mean().item(),
                }
            else:
                results[severity] = {'count': 0}
        
        return results


# ============================================================
# TEST
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Testing Metrics")
    print("=" * 60)
    
    # Create dummy predictions
    logits = torch.randn(4, 1, 64, 64)
    targets = torch.rand(4, 1, 64, 64)
    
    # Test MetricTracker
    tracker = MetricTracker(threshold=0.5)
    
    # Simulate multiple batches
    for i in range(5):
        logits = torch.randn(4, 1, 64, 64)
        targets = torch.rand(4, 1, 64, 64)
        
        # Assign categories
        categories = ['fire_1', 'fire_1', 'fire_2', 'fire_2']
        for j, cat in enumerate(categories):
            tracker.update(
                logits[j:j+1],
                targets[j:j+1],
                category=cat,
            )
    
    tracker.print_summary()
    
    # Test confusion matrix
    probs = torch.sigmoid(logits)
    cm = compute_confusion_matrix(probs, targets)
    print("\nConfusion Matrix:")
    print(f"   TP: {cm['tp']:.0f}, TN: {cm['tn']:.0f}, FP: {cm['fp']:.0f}, FN: {cm['fn']:.0f}")
    print(f"   TPR: {cm['tpr']:.4f}, TNR: {cm['tnr']:.4f}")
    
    # Test severity metrics
    severity = compute_severity_metrics(probs, targets)
    print("\nPer-Severity Metrics:")
    for sev, metrics in severity.items():
        if metrics.get('count', 0) > 0:
            print(f"   {sev}: MAE={metrics['mae']:.4f}, bias={metrics['bias']:+.4f}")
    
    print("\n[OK] All metrics working!")
    print("=" * 60)
