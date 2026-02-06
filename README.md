# EcoRevive

**AI-powered wildfire damage assessment and ecosystem restoration planning**

Transform satellite imagery into actionable restoration plans using Gemini 2.0 Flash, deep learning, and real-time Earth Engine data.

![Built with Gemini](https://img.shields.io/badge/Built%20with-Gemini%202.0-4285F4?style=for-the-badge&logo=google&logoColor=white)
![Earth Engine](https://img.shields.io/badge/Google-Earth%20Engine-34A853?style=for-the-badge&logo=google-earth&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![React](https://img.shields.io/badge/React-18-61DAFB?style=for-the-badge&logo=react&logoColor=black)

---

## The Problem

Every year, wildfires devastate millions of acres across California. After the flames subside, land managers, conservationists, and communities face a critical question: **Where do we start?**

Traditional assessment takes weeks of field surveys and expensive consulting. By then, erosion sets in and restoration windows close.

## Our Solution

EcoRevive delivers **instant, AI-powered restoration intelligence** from any location on Earth:

1. **Select any burned area** on an interactive 3D globe
2. **Get instant severity analysis** using U-Net deep learning on Sentinel-2 imagery
3. **Ask questions in natural language** — Gemini understands your context
4. **Export professional reports** ready for grants and stakeholders

---

## Key Features

### Real-Time Satellite Analysis
- **Sentinel-2 imagery** via Google Earth Engine
- **U-Net burn severity model** trained on California wildfires
- **Automatic severity classification**: High, Moderate, Low/Unburned

### Gemini-Powered Intelligence
- **Context-aware AI chat** that understands your analysis
- **Land use classification** from satellite imagery
- **Natural language recommendations** tailored to your needs
- **RAG-enhanced responses** with relevant ecosystem knowledge

### Carbon Accounting
- **IPCC Tier 2 methodology** with California-specific coefficients
- **20-year sequestration projections** with uncertainty quantification
- **Protocol eligibility** checks (Verra VCS, Gold Standard, California ARB)
- **CO₂ equivalencies**: "That's equivalent to taking X cars off the road"

### Professional Exports
- **PDF Impact Cards** for community awareness (1-2 pages)
- **Detailed Grant Reports** for professionals (5-10 pages)
- **Word documents** for stakeholder presentations
- **Chat history included** as consultation log appendix

---

## Quick Start

```bash
# 1. Clone and enter the project
git clone https://github.com/yourusername/EcoRevive.git
cd EcoRevive

# 2. Set up environment variables
echo "GOOGLE_API_KEY=your_gemini_api_key" > .env

# 3. Start the backend
cd backend
python -m venv venv
source venv/bin/activate  # Windows: .\venv\Scripts\Activate
pip install -r requirements.txt
pip install -r ../reasoning/requirements.txt
python server.py

# 4. In a new terminal, start the frontend
cd frontend
npm install
npm run dev

# 5. Open your browser
open http://localhost:5173
```

---

## Architecture

EcoRevive uses a **three-layer intelligence system**:

```
┌─────────────────────────────────────────────────────────┐
│  LAYER 1: Vision                                        │
│  ├── Google Earth Engine → Sentinel-2 imagery          │
│  └── U-Net CNN → Burn severity prediction              │
├─────────────────────────────────────────────────────────┤
│  LAYER 2: Reasoning                                     │
│  ├── Gemini 2.0 Flash → Land use classification        │
│  ├── Carbon Calculator → IPCC Tier 2 accounting        │
│  └── RAG System → Ecosystem knowledge retrieval        │
├─────────────────────────────────────────────────────────┤
│  LAYER 3: Output                                        │
│  ├── Interactive Chat → Context-aware Q&A              │
│  ├── PDF/Word Reports → Professional documentation     │
│  └── 3D Visualization → Cesium globe interface         │
└─────────────────────────────────────────────────────────┘
```

---

## Gemini Integration

EcoRevive leverages **Gemini 2.0 Flash** throughout the pipeline:

| Feature | Gemini Capability |
|---------|-------------------|
| **Land Use Analysis** | Multimodal vision on satellite imagery |
| **Interactive Chat** | Context-aware conversation with analysis memory |
| **Quick Actions** | Structured function calling for common queries |
| **Report Generation** | Narrative synthesis for professional documents |
| **Safety Assessment** | Real-time hazard identification and recommendations |

### Example Prompts
- *"Is it safe to visit this site?"* → Gets safety assessment with specific hazards
- *"What species should we plant?"* → Location-aware native species recommendations
- *"Will this qualify for carbon credits?"* → Protocol eligibility analysis

---

## Project Structure

```
EcoRevive/
├── backend/                    # FastAPI server
│   ├── server.py              # API endpoints & Gemini integration
│   └── ee_download.py         # Earth Engine data fetching
├── frontend/                   # React + Cesium UI
│   └── src/components/Globe/  # Interactive 3D globe
├── reasoning/                  # AI intelligence layer
│   ├── gemini_client.py       # Gemini API wrapper
│   ├── carbon_calculator.py   # IPCC carbon accounting
│   └── pdf_export/            # Report generation
└── California-Fire-Model/      # PyTorch U-Net model
```

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| **AI/ML** | Gemini 2.0 Flash, PyTorch, U-Net |
| **Frontend** | React 18, Vite, CesiumJS, Mapbox |
| **Backend** | FastAPI, Python 3.11 |
| **Data** | Google Earth Engine, Sentinel-2 |
| **Export** | ReportLab (PDF), python-docx (Word) |

---

## What Makes This Special

1. **End-to-end Gemini**: Not just chat — Gemini powers vision, reasoning, and generation
2. **Real satellite data**: Live Sentinel-2 imagery, not mockups
3. **Scientifically grounded**: IPCC methodology with explicit uncertainty
4. **Production-ready**: Export professional documents, not just screens
5. **Two user modes**: Personal (community) and Professional (grant-writers)



---

## License

MIT License — See [LICENSE](LICENSE) for details
