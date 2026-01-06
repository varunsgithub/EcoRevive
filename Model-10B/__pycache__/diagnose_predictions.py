#!/usr/bin/env python3
"""
Diagnostic script to identify why model predicts everything as healthy
"""

import torch
import numpy as np
from refined_model import RefinedEcoReviveModel

print("="*70)
print("🔍 EcoRevive Model Diagnostic - Finding the Issue")
print("="*70)

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n📱 Device: {device}")

# Load model
print(f"\n📦 Loading model...")
model = RefinedEcoReviveModel(device=str(device))
model.load_weights('phase2_epoch_5.pth')
model.eval()

print(f"✅ Model loaded")

# ============================================================================
# TEST 1: Check model weights are loaded correctly
# ============================================================================
print(f"\n{'='*70}")
print("TEST 1: Model Weight Statistics")
print("="*70)

total_params = 0
zero_params = 0
for name, param in model.named_parameters():
    total_params += param.numel()
    zero_params += (param == 0).sum().item()
    if 'final_conv' in name or 'outc' in name:
        print(f"\n{name}:")
        print(f"  Shape: {param.shape}")
        print(f"  Mean: {param.mean().item():.6f}")
        print(f"  Std: {param.std().item():.6f}")
        print(f"  Min: {param.min().item():.6f}")
        print(f"  Max: {param.max().item():.6f}")

print(f"\nTotal parameters: {total_params:,}")
print(f"Zero parameters: {zero_params:,} ({100*zero_params/total_params:.2f}%)")

if zero_params / total_params > 0.5:
    print("⚠️  WARNING: More than 50% of weights are zero!")

# ============================================================================
# TEST 2: Check temperature scaling parameter
# ============================================================================
print(f"\n{'='*70}")
print("TEST 2: Temperature Scaling")
print("="*70)

temp = model.temperature.temperature.item()
print(f"Temperature value: {temp:.6f}")

if temp > 10:
    print("⚠️  WARNING: Temperature is very high - this will suppress predictions!")
elif temp < 0.1:
    print("⚠️  WARNING: Temperature is very low - this may cause instability!")
else:
    print("✅ Temperature looks reasonable")

# ============================================================================
# TEST 3: Test with different input patterns
# ============================================================================
print(f"\n{'='*70}")
print("TEST 3: Input Pattern Response")
print("="*70)

test_patterns = {
    "All zeros": torch.zeros(1, 10, 256, 256),
    "All ones": torch.ones(1, 10, 256, 256),
    "Random [0,1]": torch.rand(1, 10, 256, 256),
    "All 0.5": torch.ones(1, 10, 256, 256) * 0.5,
    "High values": torch.ones(1, 10, 256, 256) * 0.9,
    "Low values": torch.ones(1, 10, 256, 256) * 0.1,
}

for name, test_input in test_patterns.items():
    test_input = test_input.to(device)
    with torch.no_grad():
        logits = model(test_input)
        probs = torch.sigmoid(logits)
    
    print(f"\n{name}:")
    print(f"  Logits - mean: {logits.mean().item():.6f}, std: {logits.std().item():.6f}")
    print(f"  Logits - min: {logits.min().item():.6f}, max: {logits.max().item():.6f}")
    print(f"  Probs  - mean: {probs.mean().item():.6f}, std: {probs.std().item():.6f}")
    print(f"  Probs  - min: {probs.min().item():.6f}, max: {probs.max().item():.6f}")

# ============================================================================
# TEST 4: Check for gradient flow issues
# ============================================================================
print(f"\n{'='*70}")
print("TEST 4: Activation Statistics (Forward Pass)")
print("="*70)

# Hook to capture activations
activations = {}

def get_activation(name):
    def hook(model, input, output):
        activations[name] = output.detach()
    return hook

# Register hooks on key layers
model.backbone.inc.register_forward_hook(get_activation('encoder_input'))
model.backbone.down4.register_forward_hook(get_activation('bottleneck'))
model.spatial_refine.register_forward_hook(get_activation('spatial_refine'))
model.final_conv.register_forward_hook(get_activation('final_conv'))

# Run forward pass
test_input = torch.rand(1, 10, 256, 256).to(device)
with torch.no_grad():
    output = model(test_input)

print("\nActivation statistics:")
for name, act in activations.items():
    print(f"\n{name}:")
    print(f"  Shape: {act.shape}")
    print(f"  Mean: {act.mean().item():.6f}")
    print(f"  Std: {act.std().item():.6f}")
    print(f"  Min: {act.min().item():.6f}")
    print(f"  Max: {act.max().item():.6f}")
    
    # Check for dead neurons
    if act.dim() == 4:  # (B, C, H, W)
        dead_channels = (act.abs().mean(dim=[0, 2, 3]) < 1e-6).sum().item()
        total_channels = act.shape[1]
        print(f"  Dead channels: {dead_channels}/{total_channels} ({100*dead_channels/total_channels:.1f}%)")

# ============================================================================
# TEST 5: Check normalization statistics
# ============================================================================
print(f"\n{'='*70}")
print("TEST 5: Normalization Check")
print("="*70)

# Simulate raw Sentinel-2 data
raw_s2 = np.random.randint(0, 10000, (10, 256, 256)).astype(np.float32)

# Apply normalization from dataset.py
BAND_MEANS = np.array([1500, 1400, 1300, 1200, 1500, 2000, 2200, 2300, 2500, 800], dtype=np.float32)
BAND_STDS = np.array([1000, 1100, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 900], dtype=np.float32)

normalized = np.clip(raw_s2, 0, 10000)
for i in range(10):
    normalized[i] = (normalized[i] - BAND_MEANS[i]) / (BAND_STDS[i] + 1e-6)
normalized = np.clip(normalized, -3, 3)
normalized = (normalized + 3) / 6

print(f"Raw data range: [{raw_s2.min():.1f}, {raw_s2.max():.1f}]")
print(f"Normalized range: [{normalized.min():.4f}, {normalized.max():.4f}]")
print(f"Normalized mean: {normalized.mean():.4f}")
print(f"Normalized std: {normalized.std():.4f}")

# Test on normalized data
norm_tensor = torch.from_numpy(normalized).float().unsqueeze(0).to(device)
with torch.no_grad():
    logits = model(norm_tensor)
    probs = torch.sigmoid(logits)

print(f"\nModel output on normalized Sentinel-2:")
print(f"  Logits - mean: {logits.mean().item():.6f}")
print(f"  Probs  - mean: {probs.mean().item():.6f}")
print(f"  Probs  - max: {probs.max().item():.6f}")

# ============================================================================
# DIAGNOSIS SUMMARY
# ============================================================================
print(f"\n{'='*70}")
print("🔍 DIAGNOSIS SUMMARY")
print("="*70)

issues_found = []

# Check 1: Low predictions
if probs.mean().item() < 0.01:
    issues_found.append("Model outputs are extremely low (< 1%)")

# Check 2: Temperature
if temp > 10:
    issues_found.append(f"Temperature is too high ({temp:.2f}) - suppressing predictions")

# Check 3: Dead neurons
if 'spatial_refine' in activations:
    act = activations['spatial_refine']
    if act.dim() == 4:
        dead_ratio = (act.abs().mean(dim=[0, 2, 3]) < 1e-6).sum().item() / act.shape[1]
        if dead_ratio > 0.5:
            issues_found.append(f"Many dead neurons ({dead_ratio*100:.1f}%)")

# Check 4: Logits range
if logits.max().item() < -5:
    issues_found.append("Logits are very negative (model is very confident in 'healthy')")

if issues_found:
    print("\n⚠️  ISSUES FOUND:")
    for i, issue in enumerate(issues_found, 1):
        print(f"  {i}. {issue}")
    
    print("\n💡 POSSIBLE SOLUTIONS:")
    if "Temperature" in str(issues_found):
        print("  • Reset temperature to 1.0")
    if "Logits are very negative" in str(issues_found):
        print("  • Model may need retraining with better class balance")
        print("  • Check if training data had correct labels")
    if "dead neurons" in str(issues_found):
        print("  • Model architecture may have issues")
        print("  • Try loading a different checkpoint")
else:
    print("\n✅ No obvious issues found in model structure")
    print("   The problem may be in the data preprocessing or training data quality")

print("\n" + "="*70)
