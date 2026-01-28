# EcoRevive 🌲🔥

AI-powered wildfire damage assessment and ecosystem restoration planning using satellite imagery, deep learning, and Gemini 3.

## Quick Start

```bash
# 1. Clone and enter the project
git clone <your-repo-url>
cd EcoRevive

# 2. Create .env file with your Gemini API key
echo "GOOGLE_API_KEY=your_key_here" > .env

# 3. Make sure you have the APIGoogKey.json file (for Earth Engine)

# 4. Start the backend
cd backend
python -m venv venv
source venv/bin/activate  # Windows: .\venv\Scripts\Activate
pip install -r requirements.txt
pip install -r ../reasoning/requirements.txt
python server.py

# 5. In a new terminal, start the frontend
cd frontend
npm install
npm run dev

# 6. Open browser
open http://localhost:5173
```

## Project Structure

```
EcoRevive/
├── backend/           # FastAPI server
│   ├── server.py      # Main API endpoints
│   └── ee_download.py # Earth Engine data fetching
├── frontend/          # React + Cesium UI
├── reasoning/         # Gemini AI integration
├── California-Fire-Model/  # PyTorch burn severity model
├── APIGoogKey.json    # Earth Engine service account (not committed)
└── .env               # API keys (not committed)
```

## Required Files (Not in Git)

| File | Description | Get From |
|------|-------------|----------|
| `.env` | Gemini API key | [AI Studio](https://aistudio.google.com/app/apikey) |
| `APIGoogKey.json` | Earth Engine service account | Project owner |

## Tech Stack

- **Frontend**: React, Vite, Cesium (3D globe)
- **Backend**: FastAPI, Python
- **AI**: Gemini 3 Flash, PyTorch
- **Data**: Google Earth Engine, Sentinel-2 imagery

---

Built for the Gemini 3 Hackathon 🚀
