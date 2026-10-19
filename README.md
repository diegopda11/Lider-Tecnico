# 🧑‍💻 Líder Técnico Virtual — Simulador de Entrevistas con IA

Aplicación en Streamlit que simula una entrevista técnica adaptativa de **3 preguntas** usando Gemini (`google-genai`) y genera un reporte final con calificación, fortalezas, áreas de mejora y dictamen (**APROBADO / NO APROBADO**).

**App desplegada:** _(pega aquí tu URL de streamlit.app)_

## Funcionalidades

- Selección de vacante: Python Junior, Frontend Web o Integrador de APIs/Bots.
- Preguntas adaptativas: cada pregunta se basa en la respuesta anterior.
- Límite estricto de 3 respuestas controlado por código (`st.session_state.turn_count`).
- Protección de rol: los intentos de desvío o jailbreak se declinan y **no cuentan** como respuesta.
- Reporte final en Markdown y chat desactivado al terminar.
- Botón **Reiniciar Entrevista** que limpia la sesión.

## Ejecutar localmente

```bash
pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # y pon tu API key
streamlit run app.py
```

## Despliegue

1. Sube el repositorio a GitHub (el `.gitignore` evita subir `secrets.toml`).
2. En [share.streamlit.io](https://share.streamlit.io) crea una app apuntando a `app.py`.
3. En **Settings → Secrets** agrega `GEMINI_API_KEY = "..."`.
