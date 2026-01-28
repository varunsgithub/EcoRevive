# EcoRevive 🌲🔥

AI-powered wildfire damage assessment and ecosystem restoration planning using satellite imagery, deep learning, and Gemini 3.

## Quick Start (Docker)

```bash
# 1. Clone and enter the project
git clone <your-repo-url>
cd EcoRevive

# 2. Make sure you have a .env file with your API key
echo "GOOGLE_API_KEY=your_key_here" > .env

# 3. Build and run everything
docker-compose up --build

# 4. Open in browser
open http://localhost:5173
```

## Development Commands

### Docker Commands

```bash
# Build and start (first time or after code changes)
docker-compose up --build

# Start without rebuilding (faster, after first build)
docker-compose up

# Run in background (detached mode)
docker-compose up -d

# View logs when running in background
docker-compose logs -f

# Stop everything
docker-compose down

# Stop and remove volumes (clean slate)
docker-compose down -v

# Rebuild just the backend
docker-compose build backend

# Shell into running backend container (for debugging)
docker exec -it ecorevive-backend-1 /bin/bash
```

### Local Development (without Docker)

**Backend:**
```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: .\venv\Scripts\Activate
pip install -r requirements.txt
pip install -r ../reasoning/requirements.txt
python server.py
```

**Frontend:**
```bash
cd frontend
npm install
npm run dev
```

## Project Structure

```
EcoRevive/
├── backend/           # FastAPI server
│   ├── server.py      # Main API endpoints
│   └── ee_download.py # Earth Engine data fetching
├── frontend/          # React + Cesium UI
├── reasoning/         # Gemini AI integration
│   ├── gemini_client.py
│   └── gemini_ecosystem.py
├── California-Fire-Model/  # PyTorch burn severity model
│   └── checkpoints/model.pth
├── Dockerfile
├── docker-compose.yml
└── .env              # Your API keys (not committed)
```

## Environment Variables

Create a `.env` file in the project root:

```
GOOGLE_API_KEY=your_gemini_api_key
```

Get your key from: https://aistudio.google.com/app/apikey

## Tech Stack

- **Frontend**: React, Vite, Cesium (3D globe)
- **Backend**: FastAPI, Python
- **AI**: Gemini 3 Flash, PyTorch
- **Data**: Google Earth Engine, Sentinel-2 imagery

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Docker build fails | Make sure Docker Desktop is running |
| Port 5173 in use | Stop other dev servers or change port in docker-compose.yml |
| Port 8000 in use | `lsof -i :8000` then `kill <PID>` |
| Gemini quota error | Wait 30 seconds between requests (free tier limit) |
| Earth Engine auth | Run `earthengine authenticate` locally first |

---

Built for the Gemini 3 Hackathon 🚀
