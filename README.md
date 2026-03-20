# Hackatón IA — COIL 2000 (Seguro de caravana)

Predicción de **interés en seguro de caravana** con el dataset [COIL 2000](https://archive.ics.uci.edu/ml/datasets/Insurance+Company+Benchmark+(COIL+2000)): modelo → API FastAPI → frontend Streamlit.

**Guía principal:** [GUIA_HACKATON.md](GUIA_HACKATON.md)

## Arranque rápido

1. Descarga los datos y colócalos en `data/` (ver [GUIA_HACKATON.md](GUIA_HACKATON.md), sección 1).
2. Crea y activa un entorno virtual; instala dependencias:
   ```bash
   cd backend && pip install -r requirements.txt
   cd ../frontend && pip install -r requirements.txt
   ```
3. Levanta la API: desde `backend/` con `uvicorn main:app --reload`, o desde la raíz con `uvicorn main:app --reload --app-dir backend`
4. Levanta Streamlit (desde `frontend/`): `streamlit run app.py`

Abre en el navegador la app (p. ej. `http://localhost:8501`) y la documentación de la API (`http://localhost:8000/docs`).
