"""
Demo: How EcoRevive cites laws in its output.

Shows the full pipeline:
1. Site conditions → Constraint checker
2. Query → RAG retrieval of relevant laws
3. Combined prompt → Gemini
4. Output with legal citations
"""

# Example of what Gemini's output would look like with legal citations:

EXPECTED_OUTPUT_EXAMPLE = """
## Restoration Recommendations for Dixie Fire Site

### Site Classification
- **Location**: 40.05°N, 121.20°W (Plumas National Forest, Wilderness Area)
- **Burn Severity**: 65% (moderate-high)
- **Ecoregion**: Sierra Nevada Mixed Conifer Forest

### ⚖️ Applicable Legal Framework

This site falls under **designated Wilderness** and is subject to:

1. **Wilderness Act of 1964** (16 U.S.C. § 1131)
   - Active restoration is generally prohibited
   - Natural regeneration should be the default approach
   - Any intervention requires Minimum Requirements Analysis

2. **National Environmental Policy Act (NEPA)**
   - Categorical Exclusion may apply for monitoring only
   - Any active intervention requires Environmental Assessment

3. **Endangered Species Act (ESA)**
   - California Spotted Owl habitat - Section 7 consultation required
   - No actions that would disturb nesting sites

### ✅ Allowed Actions (per Wilderness Act)

| Action | Status | Legal Basis |
|--------|--------|-------------|
| Natural regeneration monitoring | ✅ Allowed | Wilderness Act §4(c) |
| Invasive species removal (hand tools only) | ⚠️ Conditional | Requires MRA |
| Research and monitoring | ✅ Allowed | With permit |
| Active tree planting | ❌ Not allowed | Wilderness Act prohibition |

### 📋 Required Permits

1. **Minimum Requirements Analysis (MRA)** - For any ground-disturbing activity
2. **USFS Research Permit** - For monitoring activities
3. **ESA Section 7 Consultation** - Before any activity near spotted owl habitat

### 🌲 Recommended Approach

Given the legal constraints of Wilderness designation:

1. **Monitor natural regeneration** (no permit needed)
2. **Document seed source availability** within 500m
3. **If regeneration fails after 5 years**, submit MRA for active restoration consideration

### ⚠️ Important Notes

> Per the Wilderness Act, "wilderness areas shall be administered...so as to preserve their wilderness character" (§4(b)). Active restoration should only be considered if natural processes fail and a compelling case can be made through the MRA process.

### Uncertainty Disclosure

This recommendation is based on satellite imagery and legal database information. 
Ground-truthing by USFS personnel is required before any action. 
Confidence: 75% (moderate - specific wilderness boundary should be verified).
"""

print("=" * 70)
print("📋 EXAMPLE: How Gemini cites laws in EcoRevive output")
print("=" * 70)
print(EXPECTED_OUTPUT_EXAMPLE)
