
import numpy as np

# Simulate the user's scenario: 92% moderate severity
# Moderate threshold is 0.27 to 0.66
# Let's create an array where 92% of pixels are 0.3 (moderate)
# and 8% are 0 (unburned)

size = 1000
pixels = np.zeros(size)
num_moderate = int(0.92 * size)
pixels[:num_moderate] = 0.3

# Calculate stats using the logic from gemini_multimodal.py
mean_val = np.mean(pixels)
moderate_mask = (pixels > 0.27) & (pixels <= 0.66)
moderate_ratio = np.mean(moderate_mask)

print(f"Pixels distribution: {num_moderate} moderate, {size - num_moderate} unburned")
print(f"Mean Value: {mean_val}")
print(f"Moderate Ratio: {moderate_ratio}")

# Check if NaN behaves weirdly
pixels_nan = pixels.copy()
pixels_nan[0] = np.nan
mean_nan = np.mean(pixels_nan)
moderate_ratio_nan = np.mean((pixels_nan > 0.27) & (pixels_nan <= 0.66))

print(f"\nWith NaN:")
print(f"Mean Value: {mean_nan}")
print(f"Moderate Ratio: {moderate_ratio_nan}")
