"""
Dataset Validation Script
Checks data quality before training to catch issues early.

Validates:
- No NaN/Inf values
- Labels are in valid range (0-1)
- Spectral values are reasonable
- Sufficient valid pixels per tile
- Class balance across fires
"""

import os
import sys
import json
import numpy as np
import rasterio
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

# Add parent directory for config import
sys.path.append(str(Path(__file__).parent.parent))
from config import DATA_DIR, RAW_DATA_DIR, NUM_BANDS, SEVERITY_THRESHOLDS

# ============================================================
# CONFIGURATION
# ============================================================
DATA_PATHS = [
    RAW_DATA_DIR / "fires",
    RAW_DATA_DIR / "healthy",
]

# Validation thresholds
MIN_VALID_PIXEL_RATIO = 0.7  # At least 70% valid pixels
SPECTRAL_MIN = 0
SPECTRAL_MAX = 10000
LABEL_MIN = 0.0
LABEL_MAX = 1.0

# Output
VALIDATION_REPORT = DATA_DIR / "validation_report.json"


# ============================================================
# VALIDATION FUNCTIONS
# ============================================================
def validate_tile(tile_path):
    """
    Validate a single tile and return issues found.
    Returns: (is_valid, issues_dict, stats_dict)
    """
    issues = []
    stats = {}
    
    try:
        with rasterio.open(tile_path) as src:
            data = src.read()
            
            # Check shape
            if data.shape[0] < NUM_BANDS + 1:
                issues.append(f"Wrong band count: {data.shape[0]} (expected {NUM_BANDS + 1})")
                return False, issues, stats
            
            bands = data.shape[0]
            height, width = data.shape[1], data.shape[2]
            
            stats['shape'] = (bands, height, width)
            
            # Split spectral and label
            spectral = data[:NUM_BANDS]
            label = data[NUM_BANDS]  # 11th band is label
            
            # Check for NaN/Inf
            nan_count = np.isnan(data).sum()
            inf_count = np.isinf(data).sum()
            
            if nan_count > 0:
                issues.append(f"Contains {nan_count} NaN values")
            if inf_count > 0:
                issues.append(f"Contains {inf_count} Inf values")
            
            stats['nan_count'] = int(nan_count)
            stats['inf_count'] = int(inf_count)
            
            # Check spectral range
            spectral_valid = np.isfinite(spectral)
            spectral_in_range = (spectral >= SPECTRAL_MIN) & (spectral <= SPECTRAL_MAX)
            valid_pixels = spectral_valid & spectral_in_range
            valid_ratio = valid_pixels.all(axis=0).mean()
            
            stats['valid_pixel_ratio'] = float(valid_ratio)
            
            if valid_ratio < MIN_VALID_PIXEL_RATIO:
                issues.append(f"Low valid pixel ratio: {valid_ratio:.1%}")
            
            # Check label range
            label_valid = np.isfinite(label)
            label_in_range = (label >= LABEL_MIN) & (label <= LABEL_MAX)
            label_ok_ratio = (label_valid & label_in_range).mean()
            
            stats['label_valid_ratio'] = float(label_ok_ratio)
            
            if label_ok_ratio < 0.95:
                issues.append(f"Label issues: {(1-label_ok_ratio)*100:.1f}% out of range")
            
            # Label statistics
            valid_labels = label[label_valid & label_in_range]
            if len(valid_labels) > 0:
                stats['label_mean'] = float(valid_labels.mean())
                stats['label_std'] = float(valid_labels.std())
                stats['label_min'] = float(valid_labels.min())
                stats['label_max'] = float(valid_labels.max())
                
                # Categorize by severity
                for severity, (low, high) in SEVERITY_THRESHOLDS.items():
                    ratio = ((valid_labels >= low) & (valid_labels < high)).mean()
                    stats[f'severity_{severity}_ratio'] = float(ratio)
            
            # Spectral statistics (for first valid band as proxy)
            valid_b4 = spectral[2][valid_pixels.all(axis=0)]  # B4 (Red)
            if len(valid_b4) > 0:
                stats['spectral_mean'] = float(valid_b4.mean())
                stats['spectral_std'] = float(valid_b4.std())
            
    except Exception as e:
        issues.append(f"Read error: {str(e)}")
        return False, issues, stats
    
    is_valid = len(issues) == 0
    return is_valid, issues, stats


def collect_tiles_by_category(data_paths):
    """Collect tiles organized by fire/category."""
    tiles_by_category = defaultdict(list)
    
    for data_path in data_paths:
        if not data_path.exists():
            continue
            
        for root, dirs, files in os.walk(data_path):
            for file in files:
                if file.endswith('.tif'):
                    tile_path = Path(root) / file
                    
                    # Extract category from path
                    rel_path = tile_path.relative_to(data_path)
                    if len(rel_path.parts) >= 1:
                        category = rel_path.parts[0]  # Fire name or 'healthy'
                    else:
                        category = 'unknown'
                    
                    tiles_by_category[category].append(tile_path)
    
    return tiles_by_category


# ============================================================
# MAIN VALIDATION
# ============================================================
def main():
    print("=" * 70)
    print("🔍 DATASET VALIDATION")
    print("=" * 70)
    
    # Collect tiles
    print("\n📂 Collecting tiles by category...")
    tiles_by_category = collect_tiles_by_category(DATA_PATHS)
    
    total_tiles = sum(len(tiles) for tiles in tiles_by_category.values())
    print(f"   Found {total_tiles} tiles in {len(tiles_by_category)} categories")
    
    if total_tiles == 0:
        print("\n❌ No tiles found! Download data first.")
        return
    
    # Validate all tiles
    print("\n🔍 Validating tiles...")
    
    report = {
        'total_tiles': total_tiles,
        'valid_tiles': 0,
        'invalid_tiles': 0,
        'issues_summary': defaultdict(int),
        'categories': {},
    }
    
    invalid_files = []
    
    for category, tiles in tiles_by_category.items():
        print(f"\n   📁 {category}: {len(tiles)} tiles")
        
        cat_stats = {
            'total': len(tiles),
            'valid': 0,
            'invalid': 0,
            'label_means': [],
            'severity_distribution': defaultdict(list),
        }
        
        for tile_path in tqdm(tiles, desc=f"      Validating", leave=False):
            is_valid, issues, stats = validate_tile(tile_path)
            
            if is_valid:
                cat_stats['valid'] += 1
                report['valid_tiles'] += 1
                
                # Collect stats
                if 'label_mean' in stats:
                    cat_stats['label_means'].append(stats['label_mean'])
                    
                for key, val in stats.items():
                    if key.startswith('severity_') and key.endswith('_ratio'):
                        severity = key.replace('severity_', '').replace('_ratio', '')
                        cat_stats['severity_distribution'][severity].append(val)
            else:
                cat_stats['invalid'] += 1
                report['invalid_tiles'] += 1
                invalid_files.append({
                    'path': str(tile_path),
                    'issues': issues,
                })
                
                for issue in issues:
                    issue_type = issue.split(':')[0]
                    report['issues_summary'][issue_type] += 1
        
        # Summarize category
        if cat_stats['label_means']:
            cat_stats['avg_label'] = float(np.mean(cat_stats['label_means']))
            cat_stats['std_label'] = float(np.std(cat_stats['label_means']))
        
        # Convert severity distribution to averages
        severity_avg = {}
        for severity, ratios in cat_stats['severity_distribution'].items():
            if ratios:
                severity_avg[severity] = float(np.mean(ratios))
        cat_stats['severity_distribution'] = severity_avg
        
        cat_stats.pop('label_means', None)  # Remove raw data
        report['categories'][category] = cat_stats
    
    # Save report
    report['issues_summary'] = dict(report['issues_summary'])
    report['invalid_files'] = invalid_files[:50]  # First 50 issues
    
    print(f"\n💾 Saving report to: {VALIDATION_REPORT}")
    VALIDATION_REPORT.parent.mkdir(parents=True, exist_ok=True)
    
    with open(VALIDATION_REPORT, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 70)
    print("📊 VALIDATION SUMMARY")
    print("=" * 70)
    
    print(f"\n   Total tiles:   {report['total_tiles']}")
    print(f"   Valid tiles:   {report['valid_tiles']} ({100*report['valid_tiles']/report['total_tiles']:.1f}%)")
    print(f"   Invalid tiles: {report['invalid_tiles']} ({100*report['invalid_tiles']/report['total_tiles']:.1f}%)")
    
    if report['issues_summary']:
        print("\n   Issues found:")
        for issue, count in sorted(report['issues_summary'].items(), key=lambda x: -x[1]):
            print(f"      • {issue}: {count} tiles")
    
    # Print per-category summary
    print("\n   By Category:")
    print(f"   {'Category':<25} {'Valid':>8} {'Invalid':>8} {'Avg Label':>12}")
    print("   " + "-" * 55)
    
    for category, stats in report['categories'].items():
        avg_label = stats.get('avg_label', 0)
        print(f"   {category:<25} {stats['valid']:>8} {stats['invalid']:>8} {avg_label:>12.3f}")
    
    # Print severity distribution
    print("\n   Severity Distribution (average across valid tiles):")
    all_severities = defaultdict(list)
    for cat_stats in report['categories'].values():
        for severity, ratio in cat_stats.get('severity_distribution', {}).items():
            all_severities[severity].append(ratio)
    
    if all_severities:
        print(f"   {'Severity':<20} {'Avg Ratio':>12}")
        print("   " + "-" * 35)
        for severity in ['unburned', 'low', 'moderate_low', 'moderate_high', 'high']:
            if severity in all_severities:
                avg = np.mean(all_severities[severity])
                print(f"   {severity:<20} {avg:>12.1%}")
    
    # Final recommendation
    print("\n" + "=" * 70)
    
    if report['invalid_tiles'] == 0:
        print("✅ ALL TILES VALID! Ready for training.")
    elif report['invalid_tiles'] / report['total_tiles'] < 0.05:
        print("✅ DATASET MOSTLY CLEAN (< 5% invalid)")
        print("   Invalid tiles will be skipped during training.")
    elif report['invalid_tiles'] / report['total_tiles'] < 0.20:
        print("⚠️  SOME DATA ISSUES ({:.1%} invalid)".format(
            report['invalid_tiles'] / report['total_tiles']))
        print("   Review validation_report.json for details.")
    else:
        print("❌ SIGNIFICANT DATA ISSUES ({:.1%} invalid)".format(
            report['invalid_tiles'] / report['total_tiles']))
        print("   Check data download and reprocess if needed.")
    
    print("=" * 70)


if __name__ == "__main__":
    main()
