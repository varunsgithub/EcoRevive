# EcoRevive Architecture
## *"From Burned Land to Green Future"*

---

## What is EcoRevive?

EcoRevive is an AI-powered ecosystem restoration tool that turns satellite imagery into **actionable restoration plans**.

**The Core Insight**: Detecting degradation is only 10% of the problem. The other 90% is answering: *"What do we DO about it?"*

---

## Two Users, Two Needs

We built this system backwards—starting from what real users need, not what's technically cool.

### User 1: The Professional (NGO Project Manager)
> *"I have $500,000 and 500 hectares of burned land. Tell me exactly where to spend it for maximum impact."*

**Needs:**
- Legal/ownership data (don't get sued)
- Biological site data (what will actually grow)
- Cost-benefit prioritization (triage the budget)
- Species prescriptions (not just "plant trees")
- Monitoring framework (prove impact to donors)

### User 2: The Community Organizer
> *"The hills behind my home burned. I want to organize neighbors to help, but I don't know if it's safe or where to start."*

**Needs:**
- Safety alerts (widowmakers, landslide risk)
- "Hope Visualizer" (show the healing timeline)
- Land ownership lookup (who to call for permission)
- Simple supply checklist (what to bring)
- Shareable impact cards (recruit volunteers)

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         USER INTERFACE                               │
│              (Streamlit Dashboard / Mobile App)                      │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    LAYER 3: OUTPUT GENERATOR                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │
│  │  Report     │  │   Safety    │  │    Hope     │  │   Impact    │ │
│  │  Builder    │  │   Module    │  │  Visualizer │  │   Tracker   │ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘ │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    LAYER 2: REASONING ENGINE                         │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │                    Gemini 1.5 Pro                                ││
│  │  • Ecosystem classification (forest/wetland/grassland)          ││
│  │  • Species recommendations (native palette)                     ││
│  │  • Legal constraint checking                                    ││
│  │  • Restoration timeline forecasting                             ││
│  │  • Safety protocol generation                                   ││
│  └─────────────────────────────────────────────────────────────────┘│
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                  │
│  │  RAG:       │  │  RAG:       │  │  External   │                  │
│  │  Ecology    │  │  Legal      │  │  APIs       │                  │
│  │  Knowledge  │  │  Database   │  │  (Geocoding)│                  │
│  └─────────────┘  └─────────────┘  └─────────────┘                  │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    LAYER 1: VISION ENGINE                            │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │              California Fire Model (U-Net)                       ││
│  │              [BUILT]                                             ││
│  │                                                                  ││
│  │  Input:  10-band Sentinel-2 imagery                             ││
│  │  Output: Burn severity map (0-1 continuous)                     ││
│  │  Trained on: Dixie, Caldor, Camp, Creek fires, etc.             ││
│  └─────────────────────────────────────────────────────────────────┘│
│  ┌─────────────┐  ┌─────────────┐                                   │
│  │  NDVI       │  │  Change     │                                   │
│  │  Calculator │  │  Detection  │                                   │
│  └─────────────┘  └─────────────┘                                   │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       INPUT DATA                                     │
│  Sentinel-2 Satellite Imagery | Location Coordinates | Site Photos  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Expected Outputs (What the System Produces)

### For Professionals: The Restoration Report Package

```
EcoRevive_Report_[Site]_[Date]/
├── 01_Executive_Summary.pdf
│   └── "5 priority zones, $487K recommended spend, 12,000 tCO2 potential"
│
├── 02_Legal_Compliance/
│   ├── land_tenure.gpkg          ← Ownership boundaries
│   ├── protected_areas.gpkg      ← Parks, RAMSAR sites, etc.
│   └── legal_summary.pdf         ← "Zone A requires FS permit..."
│
├── 03_Biophysical_Data/
│   ├── burn_severity.tif         ← FROM LAYER 1 (our model!)
│   ├── slope_aspect.tif          ← Derived from DEM
│   └── site_characterization.xlsx
│
├── 04_Species_Prescriptions/
│   ├── reference_ecosystem_map.gpkg
│   ├── species_palette.xlsx      ← "Plant Ponderosa Pine, not Eucalyptus"
│   └── planting_design.pdf
│
├── 05_Prioritization/
│   ├── priority_zones.gpkg       ← Ranked by impact/cost
│   └── cost_benefit_matrix.xlsx
│
└── 06_Monitoring_Framework/
    ├── baseline_ndvi.csv
    └── monitoring_protocol.pdf
```

### For Community: The Mobile App Screens

| Screen | What It Shows | Source |
|--------|---------------|--------|
| **Safety Check** | Red zones (widowmakers), landslide risk | Layer 2 (Gemini) |
| **Land Ownership** | "US Forest Service, call Jennifer at 530-555-0142" | Layer 2 (geocoding API) |
| **Hope Visualizer** | Slider: Now → Year 15 with AI-rendered recovery | Layer 3 (image gen) |
| **Supply Checklist** | "50 straw wattles, 20 lbs native seed mix" | Layer 2 (Gemini) |
| **Impact Card** | Shareable "We planted 200 trees today!" graphic | Layer 3 |

---

## Component Details

### Layer 1: Vision Engine (BUILT)

**File:** `California-Fire-Model/`

| Component | Status | Description |
|-----------|--------|-------------|
| `model/architecture.py` | Done | U-Net with attention gates |
| `config.py` | Done | 7 training fires, 2 test fires |
| `data/dataset.py` | Done | Sentinel-2 10-band loader |
| `inference/predict.py` | Done | Run model on new imagery |

**What it produces:**
```python
# Input: 10-band Sentinel-2 image (B, 10, 256, 256)
# Output: Burn severity probability map (B, 1, 256, 256)
#         Values 0.0 (healthy) → 1.0 (severely burned)
```

---

### Layer 2: Reasoning Engine (TO BUILD)

**Purpose:** Take vision outputs + location and generate restoration intelligence.

#### Module 2A: `gemini_ecosystem.py`
```python
def classify_ecosystem(location: tuple, burn_severity_map: np.array) -> dict:
    """
    Given a location and our model's output, identify:
    - Reference ecosystem (what was here before?)
    - Appropriate native species palette
    - Restoration method (reforestation vs natural regeneration)
    """
```

#### Module 2B: `gemini_safety.py`
```python
def generate_safety_report(location: tuple, severity_map: np.array, slope_map: np.array) -> dict:
    """
    Generate safety warnings:
    - Widowmaker zones (high severity + standing dead trees)
    - Landslide risk (steep slopes + recent burn)
    - Access restrictions
    """
```

#### Module 2C: `gemini_legal.py`
```python
def check_land_ownership(location: tuple) -> dict:
    """
    Query land ownership databases:
    - Federal (Forest Service, BLM, NPS)
    - State parks
    - Private (with contact info if available)
    - Indigenous territories (flag for FPIC)
    """
```

#### Module 2D: `gemini_hope.py`
```python
def forecast_recovery(ecosystem_type: str, severity: float, intervention: str, year: int) -> dict:
    """
    Generate recovery timeline:
    - Year 1: Grass coverage %
    - Year 5: Shrub establishment
    - Year 15: Forest canopy %
    - Carbon sequestration estimates
    """
```

---

### Layer 3: Output Generator (TO BUILD)

**Purpose:** Format Layer 2 outputs for consumption.

| Module | Professional Output | Community Output |
|--------|---------------------|------------------|
| `report_builder.py` | PDF + Shapefiles | - |
| `app_screens.py` | Dashboard view | Mobile-friendly screens |
| `visualizer.py` | GIS maps | Before/After slider |
| `impact_tracker.py` | Monitoring dashboard | Shareable impact cards |

---

## Data Flow Example

### Scenario: Scanning Blackwood Ridge (100 hectares, burned in 2025)

```
Step 1: INPUT
├── User uploads Sentinel-2 tile or enters coordinates
└── Location: (39.42, -121.15), Butte County, CA

Step 2: LAYER 1 (Vision)
├── California Fire Model processes 10-band image
├── Output: burn_severity.tif (0-1 continuous)
└── Result: 78% of area > 0.6 severity ("High")

Step 3: LAYER 2 (Reasoning)
├── gemini_ecosystem.py
│   └── "Reference ecosystem: Mixed Conifer Forest"
│   └── "Species: Ponderosa Pine, Black Oak, Manzanita"
├── gemini_safety.py
│   └── "47 widowmaker hazard zones detected"
│   └── "3 landslide-risk slopes"
├── gemini_legal.py
│   └── "Tahoe National Forest (USFS)"
│   └── "Contact: Jennifer Walsh, (530) 555-0142"
└── gemini_hope.py
    └── "Year 5: 40% ground cover | Year 15: Young forest"

Step 4: LAYER 3 (Output)
├── PROFESSIONAL: Full report package (PDF + GIS)
└── COMMUNITY: Safety screen + Hope Visualizer + Supply list
```

---

## Implementation Priority

### Phase 1: Connect Vision to Reasoning (Week 1)
- [ ] Create `gemini_ecosystem.py` - classify ecosystem from location
- [ ] Create `gemini_safety.py` - generate safety protocols
- [ ] Test with 3 sites from our training data

### Phase 2: Add Legal & Hope (Week 2)
- [ ] Create `gemini_legal.py` - land ownership lookup
- [ ] Create `gemini_hope.py` - recovery forecasting
- [ ] Build basic Streamlit dashboard

### Phase 3: Professional Outputs (Week 3)
- [ ] Report builder (PDF generation)
- [ ] GIS export (Shapefiles/GeoPackage)
- [ ] Cost-benefit matrix generator

### Phase 4: Community Features (Week 4)
- [ ] Hope Visualizer (before/after slider)
- [ ] Safety briefing generator
- [ ] Impact card creator

---

## Key Insight

**We already have the hardest part done.**

The California Fire Model is the "eyes" of EcoRevive. Now we need to add:
1. **The brain** (Gemini reasoning)
2. **The voice** (output generation)

The remaining work is integration, not new ML training.

---

## Proposed Directory Structure

```
EcoRevive/
├── California-Fire-Model/     <- EXISTING (Vision Layer)
│   ├── model/
│   ├── inference/
│   └── ...
│
├── reasoning/                  <- NEW (Layer 2)
│   ├── gemini_ecosystem.py
│   ├── gemini_safety.py
│   ├── gemini_legal.py
│   ├── gemini_hope.py
│   └── knowledge_base/
│       ├── ecology_prompts.md
│       └── legal_rules.json
│
├── outputs/                    <- NEW (Layer 3)
│   ├── report_builder.py
│   ├── visualizer.py
│   ├── impact_tracker.py
│   └── templates/
│
├── frontend/                   <- NEW (UI)
│   └── app.py                 # Streamlit dashboard
│
└── ARCHITECTURE.md            <- YOU ARE HERE
```

---

## Success Criteria

**For Professionals:**
> "I can use this report to allocate my $500K budget and defend my decisions to donors."

**For Community:**
> "I feel safe bringing volunteers to this site and I can show them what we're building toward."

---

*Built by the EcoRevive Team*
