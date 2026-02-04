# EcoRevive: Windows Setup Guide

---

## Option A: Docker (Recommended - Easiest!)

If you have Docker Desktop installed, this is the simplest way:

```powershell
# Just run this ONE command:
docker-compose up --build
```

Then open **http://localhost:5173** in your browser. Done!

> Install Docker Desktop from: https://www.docker.com/products/docker-desktop/

---

## Option B: Manual Setup (If you don't want Docker)

1.  **Python 3.10+**: Download from [python.org](https://www.python.org/downloads/). During install, **check "Add Python to PATH"**.
2.  **Node.js 18+**: Download from [nodejs.org](https://nodejs.org/).
3.  **Git**: Download from [git-scm.com](https://git-scm.com/download/win).

---

## Step 1: Clone the Repository

Open **PowerShell** or **Command Prompt** and run:

```powershell
git clone <your-repo-url>
cd EcoRevive
```

---

## Step 2: Set Up the Python Backend

```powershell
# Navigate to backend
cd backend

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
.\venv\Scripts\Activate

# Install dependencies
pip install -r requirements.txt

# Also install reasoning dependencies
pip install -r ../reasoning/requirements.txt
```

> [!WARNING]
> **PyTorch on Windows**: The default `pip install torch` might not work. If you have an NVIDIA GPU, install with CUDA:
> ```powershell
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
> ```
> If you have no GPU (CPU only):
> ```powershell
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
> ```

---

## Step 3: Set Up Environment Variables

Create a `.env` file in the project root (`EcoRevive/.env`):

```
GOOGLE_API_KEY=your_gemini_api_key_here
```

Get your API key from [Google AI Studio](https://aistudio.google.com/app/apikey).

---

## Step 4: Authenticate Earth Engine

```powershell
# Still in backend folder with venv activated
python -c "import ee; ee.Authenticate()"
```

This will open a browser window. Log in with your Google account and paste the code back into the terminal.

---

## Step 5: Set Up the Frontend

Open a **new terminal** (keep the backend one open):

```powershell
cd frontend
npm install
```

---

## Step 6: Run the Application

**Terminal 1 (Backend):**
```powershell
cd backend
.\venv\Scripts\Activate
python server.py
```

**Terminal 2 (Frontend):**
```powershell
cd frontend
npm run dev
```

Open your browser to: **http://localhost:5173**

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `python` not found | Use `python3` or reinstall Python with "Add to PATH" checked |
| `venv\Scripts\Activate` fails | Run `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned` in PowerShell |
| `torch` import error | Reinstall PyTorch using the commands above |
| `httptools` build error | Already fixed! We removed `uvicorn[standard]`. If still failing, run: `pip install uvicorn --no-deps` then `pip install click h11 typing-extensions` |
| Earth Engine auth fails | Make sure you have a Google Cloud project with Earth Engine API enabled |
| Port 8000 in use | Kill the process: `netstat -ano | findstr :8000` then `taskkill /PID <pid> /F` |

---

## Quick Start Checklist

- [ ] Python 3.10+ installed
- [ ] Node.js 18+ installed
- [ ] Git installed
- [ ] Repository cloned
- [ ] Backend `venv` created and activated
- [ ] `pip install -r requirements.txt` completed
- [ ] PyTorch installed correctly
- [ ] `.env` file created with `GOOGLE_API_KEY`
- [ ] Earth Engine authenticated
- [ ] `npm install` completed in frontend
- [ ] Backend running on port 8000
- [ ] Frontend running on port 5173
