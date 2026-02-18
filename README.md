# Smart Turn — Streamlit Deployment

## 📁 Files in this folder

| File | Purpose |
|------|---------|
| `app.py` | Main Streamlit app (UI + logic) |
| `model_service.py` | Turn-detection model (SemanticAnalyzer) |
| `asr_service.py` | Speech-to-text services (Groq / Local Whisper) |
| `requirements.txt` | Python dependencies |
| `.env` | API keys (local) |
| `.streamlit/secrets.toml` | API keys (Streamlit Cloud) |
| `models/turn_detection_model/` | Fine-tuned model files (**you must copy this**) |

## 🚀 How to Deploy to Streamlit Cloud

### Step 1: Copy the model
Copy the `models/turn_detection_model` folder from the project root into this `streamlit` folder:

```
Smart Turn/
├── models/turn_detection_model/   ← source
└── streamlit/
    └── models/turn_detection_model/   ← copy here
```

### Step 2: Push to GitHub
Create a new GitHub repo and push **only the contents of this `streamlit` folder** as the root of that repo:

```
your-repo/
├── app.py
├── model_service.py
├── asr_service.py
├── requirements.txt
├── .streamlit/secrets.toml   (or add secrets via dashboard)
└── models/turn_detection_model/
```

### Step 3: Deploy on Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Connect your GitHub repo
3. Set **Main file path** to `app.py`
4. Go to **Advanced Settings → Secrets** and add:
   ```
   GROQ_API_KEY = "your_groq_api_key_here"
   ```
5. Click **Deploy**

## 🖥️ Run Locally

```bash
cd streamlit
pip install -r requirements.txt
streamlit run app.py
```

## 📝 Features
- **Audio Analysis**: Record or upload a WAV file → VAD → ASR → Turn Detection
- **Text Analysis**: Type text and check if the sentence is complete or incomplete
