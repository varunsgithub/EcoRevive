"""
Hard Negative Mining for EcoRevive Model
Identifies difficult training examples to focus training on failed cases
"""

import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import json


class HardNegativeMiner:
    """
    Mines hard examples from dataset using trained model
    Ranks samples by prediction error and uncertainty
    """
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.model.eval()
    
    def compute_difficulty_scores(self, dataset, batch_size=8, num_workers=2):
        """
        Compute difficulty score for each sample in dataset
        
        Returns:
            difficulty_scores: dict mapping index -> score (0=easy, 1=hard)
            sample_metrics: detailed metrics per sample
        """
        print(f"Computing difficulty scores for {len(dataset)} samples...")
        
        # Create dataloader
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        difficulty_scores = {}
        sample_metrics = {}
        
        current_idx = 0
        
        with torch.no_grad():
            for images, masks in tqdm(loader, desc="Mining hard examples"):
                images = images.to(self.device)
                masks = masks.unsqueeze(1).to(self.device)
                
                # Forward pass
                logits = self.model(images)
                probs = torch.sigmoid(logits)
                
                # Compute metrics per sample
                for i in range(images.shape[0]):
                    sample_idx = current_idx + i
                    
                    pred = probs[i, 0].cpu().numpy()
                    target = masks[i, 0].cpu().numpy()
                    
                    # Metrics for difficulty
                    mae = np.abs(pred - target).mean()
                    mse = ((pred - target) ** 2).mean()
                    
                    # IoU error
                    intersection = (pred * target).sum()
                    union = pred.sum() + target.sum() - intersection
                    iou = intersection / (union + 1e-6)
                    iou_error = 1.0 - iou
                    
                    # Uncertainty (distance from 0 or 1)
                    uncertainty = (1.0 - 2.0 * np.abs(pred - 0.5)).mean()
                    
                    # Combine into difficulty score
                    # High difficulty = high error + high uncertainty
                    difficulty = (
                        0.4 * mae +           # Mean absolute error
                        0.3 * iou_error +     # IoU error
                        0.3 * uncertainty     # Prediction uncertainty
                    )
                    
                    difficulty_scores[sample_idx] = float(difficulty)
                    sample_metrics[sample_idx] = {
                        'mae': float(mae),
                        'mse': float(mse),
                        'iou': float(iou),
                        'iou_error': float(iou_error),
                        'uncertainty': float(uncertainty),
                        'difficulty': float(difficulty),
                        'mean_pred': float(pred.mean()),
                        'mean_target': float(target.mean()),
                    }
                
                current_idx += images.shape[0]
        
        return difficulty_scores, sample_metrics
    
    def get_hard_examples(self, difficulty_scores, top_k=None, percentile=70):
        """
        Get indices of hardest examples
        
        Args:
            difficulty_scores: dict from compute_difficulty_scores
            top_k: return top K hardest examples (if None, use percentile)
            percentile: return top X percentile (default: top 30%)
        
        Returns:
            hard_indices: list of hard example indices
        """
        scores = np.array(list(difficulty_scores.values()))
        indices = np.array(list(difficulty_scores.keys()))
        
        if top_k is not None:
            # Get top K
            hard_idx = np.argsort(scores)[-top_k:]
        else:
            # Get top percentile
            threshold = np.percentile(scores, percentile)
            hard_idx = np.where(scores >= threshold)[0]
        
        hard_indices = indices[hard_idx].tolist()
        
        print(f"Selected {len(hard_indices)} hard examples")
        print(f"  Difficulty range: [{scores[hard_idx].min():.4f}, {scores[hard_idx].max():.4f}]")
        print(f"  Average difficulty: {scores[hard_idx].mean():.4f}")
        
        return hard_indices
    
    def categorize_by_difficulty(self, difficulty_scores):
        """
        Categorize samples into easy/medium/hard
        
        Returns:
            easy_indices, medium_indices, hard_indices
        """
        scores = np.array(list(difficulty_scores.values()))
        indices = np.array(list(difficulty_scores.keys()))
        
        # Percentile-based categorization
        p33 = np.percentile(scores, 33)
        p66 = np.percentile(scores, 66)
        
        easy_idx = np.where(scores < p33)[0]
        medium_idx = np.where((scores >= p33) & (scores < p66))[0]
        hard_idx = np.where(scores >= p66)[0]
        
        print(f"Difficulty distribution:")
        print(f"  Easy: {len(easy_idx)} samples (difficulty < {p33:.4f})")
        print(f"  Medium: {len(medium_idx)} samples ({p33:.4f} - {p66:.4f})")
        print(f"  Hard: {len(hard_idx)} samples (difficulty >= {p66:.4f})")
        
        return (
            indices[easy_idx].tolist(),
            indices[medium_idx].tolist(),
            indices[hard_idx].tolist()
        )
    
    def save_results(self, difficulty_scores, sample_metrics, output_path):
        """Save mining results to JSON"""
        results = {
            'difficulty_scores': difficulty_scores,
            'sample_metrics': sample_metrics,
            'statistics': {
                'num_samples': len(difficulty_scores),
                'mean_difficulty': float(np.mean(list(difficulty_scores.values()))),
                'std_difficulty': float(np.std(list(difficulty_scores.values()))),
                'min_difficulty': float(np.min(list(difficulty_scores.values()))),
                'max_difficulty': float(np.max(list(difficulty_scores.values()))),
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"✅ Saved results to {output_path}")


# Example usage
if __name__ == "__main__":
    from refined_model import RefinedEcoReviveModel
    from dataset import EcoReviveDataset
    
    print("="*70)
    print("🔍 Hard Negative Mining Example")
    print("="*70)
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RefinedEcoReviveModel(device=str(device))
    model.load_weights('phase2_epoch_5_calibrated_v2.pth')
    model.eval()
    
    # Create dataset (replace with your data paths)
    data_dirs = [
        "/path/to/your/data/EcoRevive_Balanced-Forest",
        # Add more directories
    ]
    
    # Uncomment to run
    # dataset = EcoReviveDataset(data_dirs, augment=False, mode='val')
    
    # miner = HardNegativeMiner(model, device)
    # difficulty_scores, sample_metrics = miner.compute_difficulty_scores(dataset)
    
    # # Get hard examples
    # hard_indices = miner.get_hard_examples(difficulty_scores, percentile=70)
    
    # # Categorize
    # easy, medium, hard = miner.categorize_by_difficulty(difficulty_scores)
    
    # # Save
    # miner.save_results(difficulty_scores, sample_metrics, 'hard_examples.json')
    
    print("\n✅ Example complete (comment out to run with real data)")
    print("="*70)
