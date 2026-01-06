#!/usr/bin/env python3
"""
Fix for model prediction bias - applies logit calibration
"""

import torch
import numpy as np
from refined_model import RefinedEcoReviveModel

print("="*70)
print("🔧 EcoRevive Model - Prediction Bias Fix")
print("="*70)

# The diagnostic showed that logits are around -6 to -15
# This means the model is heavily biased toward "healthy" (class 0)
# 
# SOLUTION: Apply a bias correction to the final layer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model
print(f"\n📦 Loading model...")
model = RefinedEcoReviveModel(device=str(device))
model.load_weights('phase2_epoch_5.pth')
model.eval()

print(f"✅ Model loaded")

# Check current bias
print(f"\n🔍 Current model state:")
print(f"   Final conv bias: {model.final_conv.bias.item():.6f}")
print(f"   Temperature: {model.temperature.temperature.item():.6f}")

# Test current predictions
test_input = torch.rand(1, 10, 256, 256).to(device)
with torch.no_grad():
    logits_before = model(test_input)
    probs_before = torch.sigmoid(logits_before)

print(f"\n📊 Before calibration:")
print(f"   Logits mean: {logits_before.mean().item():.4f}")
print(f"   Probs mean: {probs_before.mean().item():.6f}")
print(f"   Probs max: {probs_before.max().item():.6f}")

# SOLUTION 1: Add positive bias to final conv layer
# This shifts all logits upward, making degraded predictions more likely
BIAS_CORRECTION = 6.0  # Add +6 to logits (shifts sigmoid from ~0.002 to ~0.5)

print(f"\n🔧 Applying bias correction: +{BIAS_CORRECTION}")
model.final_conv.bias.data += BIAS_CORRECTION

# Test after correction
with torch.no_grad():
    logits_after = model(test_input)
    probs_after = torch.sigmoid(logits_after)

print(f"\n📊 After calibration:")
print(f"   Logits mean: {logits_after.mean().item():.4f}")
print(f"   Probs mean: {probs_after.mean().item():.6f}")
print(f"   Probs max: {probs_after.max().item():.6f}")

# Save calibrated model
output_path = 'phase2_epoch_5_calibrated.pth'
print(f"\n💾 Saving calibrated model to: {output_path}")

# Save the full model state
torch.save({
    'model_state_dict': model.state_dict(),
    'bias_correction': BIAS_CORRECTION,
    'temperature': model.temperature.temperature.item(),
}, output_path)

print(f"✅ Calibrated model saved!")

print(f"\n📝 Usage:")
print(f"   1. Update misc.ipynb to use 'phase2_epoch_5_calibrated.pth'")
print(f"   2. Or manually add this line after loading the model:")
print(f"      model.final_conv.bias.data += {BIAS_CORRECTION}")

print("\n" + "="*70)
print("✅ Calibration complete!")
print("="*70)
