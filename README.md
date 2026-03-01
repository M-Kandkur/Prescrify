# 💊 Prescrify

> 🚀 AMD Slingshot 2026 | AI-powered prescription analyzer that finds cheaper generic alternatives, detects medical bill overcharges and saves money for 1.4 billion Indians — powered by CrewAI + Groq + Llama 3.3 70B on 253,973 real Indian medicines 🇮🇳💊

## ✨ Features
- 🔍 **Medicine Search** — Search 253,973 real Indian medicines
- 📄 **Prescription Analyzer** — Upload prescription image → AI extracts medicines
- 🧾 **Bill Overcharge Detector** — Detect overcharges in medical bills instantly
- 💰 **Savings Dashboard** — See how much you save by switching to generics
- 🏪 **Nearby Store Locator** — Find nearest Jan Aushadhi stores

## 🛠️ Tech Stack
- **CrewAI** — Multi-agent AI framework
- **Groq** — Free LLM API (Llama 3.3 70B)
- **Streamlit** — Web UI
- **EasyOCR** — Prescription image reading
- **Indian Medicine Dataset** — 253,973 NPPA verified medicines

## ▶️ Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## 🌐 Deploy on Streamlit Cloud
1. Fork this repo
2. Go to share.streamlit.io
3. Connect repo → Deploy
4. Add `GROQ_API_KEY` in Secrets

## 🏆 Built For
AMD Slingshot Hackathon 2026

## ⚠️ Disclaimer
Always consult your doctor before switching medicines.

## 📊 Stats
- 253,973 Indian medicines indexed
- Covers 1.4 billion Indians
- Powered by Llama 3.3 70B