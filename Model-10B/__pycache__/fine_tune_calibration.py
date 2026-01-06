#!/usr/bin/env python3
"""
Fine-tune calibration to increase sensitivity to degradation
Based on user feedback that model is too conservative
"""

import torch
import numpy as np
from refined_model import RefinedEcoReviveModel

print("="*70)
print("🔧 Fine-Tuning Model Calibration")
print("="*70)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the current calibrated model
print(f"\n📦 Loading calibrated model...")
model = RefinedEcoReviveModel(device=str(device))
model.load_weights('phase2_epoch_5_calibrated.pth')
model.eval()

print(f"✅ Model loaded")

# Check current state
current_bias = model.final_conv.bias.item()
print(f"\n🔍 Current bias: {current_bias:.6f}")

# Test with different calibration levels
test_input = torch.rand(1, 10, 256, 256).to(device)

print(f"\n📊 Testing different calibration levels:")
print(f"{'Bias Correction':<20} {'Mean Prob':<15} {'Max Prob':<15} {'% > 0.5':<15}")
print("-" * 70)

calibration_levels = [6.0, 7.0, 8.0, 9.0, 10.0]
results = {}

for additional_bias in calibration_levels:
    # Reset to original calibrated state
    model.load_weights('phase2_epoch_5_calibrated.pth')
    
    # Add additional bias (on top of the +6.0 already in calibrated model)
    extra = additional_bias - 6.0
    if extra != 0:
        model.final_conv.bias.data += extra
    
    with torch.no_grad():
        logits = model(test_input)
        probs = torch.sigmoid(logits).cpu().numpy()[0, 0]
    
    mean_prob = probs.mean()
    max_prob = probs.max()
    pct_degraded = (probs > 0.5).mean() * 100
    
    results[additional_bias] = {
        'mean': mean_prob,
        'max': max_prob,
        'pct_degraded': pct_degraded
    }
    
    print(f"+{additional_bias:<19.1f} {mean_prob:<15.4f} {max_prob:<15.4f} {pct_degraded:<15.1f}%")

# Recommendation
print(f"\n{'='*70}")
print("💡 RECOMMENDATIONS")
print("="*70)

print(f"\n📌 Current (+6.0):")
print(f"   - Conservative: Good for avoiding false positives")
print(f"   - May miss some degraded areas")

print(f"\n📌 Recommended (+8.0):")
print(f"   - Balanced: Better sensitivity to degradation")
print(f"   - Should highlight more deforested areas")
print(f"   - Mean prob: {results[8.0]['mean']:.4f}")
print(f"   - Degraded pixels: {results[8.0]['pct_degraded']:.1f}%")

print(f"\n📌 Aggressive (+10.0):")
print(f"   - High sensitivity: Catches all potential degradation")
print(f"   - May have more false positives")
print(f"   - Mean prob: {results[10.0]['mean']:.4f}")
print(f"   - Degraded pixels: {results[10.0]['pct_degraded']:.1f}%")

# Ask user which to use
print(f"\n{'='*70}")
print("🎯 CREATE NEW CALIBRATED MODEL?")
print("="*70)
print(f"\nOptions:")
print(f"  1. Keep current (+6.0) - Conservative")
print(f"  2. Use recommended (+8.0) - Balanced ⭐")
print(f"  3. Use aggressive (+10.0) - High sensitivity")
print(f"  4. Custom value")

choice = input(f"\nEnter choice (1-4) [default: 2]: ").strip() or "2"

if choice == "1":
    print(f"\n✅ Keeping current calibration (+6.0)")
elif choice == "2":
    bias_value = 8.0
    print(f"\n🔧 Creating model with +{bias_value} calibration...")
    
    # Load original and apply new calibration
    model.load_weights('phase2_epoch_5.pth')
    model.final_conv.bias.data += bias_value
    
    # Save
    output_path = 'phase2_epoch_5_calibrated_v2.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'bias_correction': bias_value,
        'temperature': model.temperature.temperature.item(),
    }, output_path)
    
    print(f"✅ Saved to: {output_path}")
    print(f"\n📝 To use in misc.ipynb:")
    print(f"   model_path = 'phase2_epoch_5_calibrated_v2.pth'")
    
elif choice == "3":
    bias_value = 10.0
    print(f"\n🔧 Creating model with +{bias_value} calibration...")
    
    model.load_weights('phase2_epoch_5.pth')
    model.final_conv.bias.data += bias_value
    
    output_path = 'phase2_epoch_5_calibrated_aggressive.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'bias_correction': bias_value,
        'temperature': model.temperature.temperature.item(),
    }, output_path)
    
    print(f"✅ Saved to: {output_path}")
    print(f"\n📝 To use in misc.ipynb:")
    print(f"   model_path = 'phase2_epoch_5_calibrated_aggressive.pth'")
    
elif choice == "4":
    try:
        bias_value = float(input("Enter custom bias value (e.g., 7.5): "))
        print(f"\n🔧 Creating model with +{bias_value} calibration...")
        
        model.load_weights('phase2_epoch_5.pth')
        model.final_conv.bias.data += bias_value
        
        output_path = f'phase2_epoch_5_calibrated_custom.pth'
        torch.save({
            'model_state_dict': model.state_dict(),
            'bias_correction': bias_value,
            'temperature': model.temperature.temperature.item(),
        }, output_path)
        
        print(f"✅ Saved to: {output_path}")
    except ValueError:
        print("❌ Invalid value")

print("\n" + "="*70)
