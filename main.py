from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from scalar_fastapi import get_scalar_api_reference
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

app = FastAPI(
    title="API de Predicción de Flores",
    version="2.1.0",
    docs_url="/swagger",
    redoc_url=None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modelo entrenado al iniciar con el dataset Iris
iris = load_iris()
model = RandomForestClassifier(n_estimators=150, random_state=42)
model.fit(iris.data, iris.target)

ESPECIES = {
    0: {
        "nombre": "Iris Setosa",
        "descripcion": "Pétalos cortos y anchos; suele separarse con claridad de las otras especies.",
    },
    1: {
        "nombre": "Iris Versicolor",
        "descripcion": "Medidas intermedias en sépalos y pétalos respecto a Setosa y Virginica.",
    },
    2: {
        "nombre": "Iris Virginica",
        "descripcion": "Pétalos y sépalos generalmente más largos y anchos en el conjunto Iris.",
    },
}


class FlorInput(BaseModel):
    sepal_length: float = Field(..., gt=0, le=20, description="Largo del sépalo en cm")
    sepal_width: float = Field(..., gt=0, le=20, description="Ancho del sépalo en cm")
    petal_length: float = Field(..., gt=0, le=20, description="Largo del pétalo en cm")
    petal_width: float = Field(..., gt=0, le=20, description="Ancho del pétalo en cm")


@app.post("/predict", summary="Predecir Flor")
def predecir_flor(flor: FlorInput):
    x = [[
        flor.sepal_length,
        flor.sepal_width,
        flor.petal_length,
        flor.petal_width,
    ]]
    pred_idx = int(model.predict(x)[0])
    probs = model.predict_proba(x)[0]

    probabilidades = {
        ESPECIES[i]["nombre"]: round(float(p) * 100, 2)
        for i, p in enumerate(probs)
    }

    return {
        "flor": ESPECIES[pred_idx]["nombre"],
        "clase_id": pred_idx,
        "confianza_porcentaje": probabilidades[ESPECIES[pred_idx]["nombre"]],
        "descripcion": ESPECIES[pred_idx]["descripcion"],
        "probabilidades": probabilidades,
        "parametros": flor.model_dump(),
    }


@app.get("/api-info")
def api_info():
    return {
        "app": "API de Predicción de Flores",
        "version": "2.1.0",
        "web": "/",
        "predict": "/predict",
        "docs": "/docs",
        "swagger": "/swagger",
        "openapi": "/openapi.json",
    }


@app.get("/docs", include_in_schema=False)
def scalar_docs():
    return get_scalar_api_reference(
        openapi_url=app.openapi_url,
        title=f"{app.title} - Scalar",
    )


WEB_HTML = """<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Clasificador de Flores — FastAPI + ML</title>
  <style>
    :root {
      --bg: #0b1120;
      --panel: #111827;
      --panel-2: #1e293b;
      --border: #334155;
      --text: #f8fafc;
      --muted: #94a3b8;
      --accent: #10b981;
      --accent-hover: #059669;
      --blue: #38bdf8;
    }
    * { box-sizing: border-box; font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; }
    body {
      margin: 0;
      min-height: 100vh;
      background: radial-gradient(circle at top, #172554 0%, var(--bg) 55%);
      color: var(--text);
      padding: 24px 16px;
    }
    .wrap {
      max-width: 980px;
      margin: 0 auto;
    }
    header {
      display: flex;
      flex-wrap: wrap;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      margin-bottom: 20px;
      padding: 18px 22px;
      background: rgba(17, 24, 39, 0.9);
      border: 1px solid var(--border);
      border-radius: 16px;
    }
    header h1 { margin: 0; font-size: 1.35rem; }
    header p { margin: 4px 0 0; color: var(--muted); font-size: 0.9rem; }
    .links { display: flex; gap: 10px; flex-wrap: wrap; }
    .links a {
      color: var(--blue);
      text-decoration: none;
      font-size: 0.85rem;
      padding: 6px 12px;
      border: 1px solid var(--border);
      border-radius: 8px;
      background: var(--panel-2);
    }
    .links a:hover { border-color: var(--blue); }
    .grid {
      display: grid;
      grid-template-columns: 1.05fr 0.95fr;
      gap: 20px;
    }
    @media (max-width: 800px) {
      .grid { grid-template-columns: 1fr; }
    }
    .card {
      background: rgba(17, 24, 39, 0.92);
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 20px;
    }
    .card h2 { margin: 0 0 6px; font-size: 1.1rem; }
    .sub { color: var(--muted); font-size: 0.85rem; margin-bottom: 16px; }
    .presets { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 14px; }
    .preset-btn {
      background: var(--panel-2);
      color: var(--text);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 6px 10px;
      font-size: 0.8rem;
      cursor: pointer;
    }
    .preset-btn:hover { border-color: var(--accent); }
    .inputs {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
    }
    label { display: block; font-size: 0.82rem; color: #cbd5e1; margin-bottom: 5px; }
    input {
      width: 100%;
      padding: 10px 12px;
      border-radius: 10px;
      border: 1px solid var(--border);
      background: var(--bg);
      color: var(--text);
      font-size: 0.95rem;
    }
    input:focus { outline: none; border-color: var(--accent); }
    .btn-predict {
      width: 100%;
      margin-top: 16px;
      padding: 12px;
      border: none;
      border-radius: 10px;
      background: var(--accent);
      color: #06281e;
      font-weight: 700;
      font-size: 0.95rem;
      cursor: pointer;
    }
    .btn-predict:hover { background: var(--accent-hover); color: #fff; }
    .result-title {
      font-size: 1.4rem;
      font-weight: 800;
      color: #34d399;
      margin: 4px 0 6px;
    }
    .desc { color: #cbd5e1; font-size: 0.9rem; margin-bottom: 14px; }
    .prob-row { margin-bottom: 10px; }
    .prob-label {
      display: flex;
      justify-content: space-between;
      font-size: 0.84rem;
      margin-bottom: 4px;
    }
    .bar-bg {
      height: 8px;
      background: var(--bg);
      border-radius: 999px;
      overflow: hidden;
    }
    .bar-fill {
      height: 100%;
      background: linear-gradient(90deg, #10b981, #38bdf8);
      border-radius: 999px;
      transition: width 0.25s ease;
    }
    .history-top {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 12px;
    }
    .btn-clear {
      background: #450a0a;
      color: #fca5a5;
      border: 1px solid #7f1d1d;
      border-radius: 8px;
      padding: 6px 10px;
      font-size: 0.78rem;
      cursor: pointer;
    }
    .history-list {
      display: flex;
      flex-direction: column;
      gap: 10px;
      max-height: 360px;
      overflow-y: auto;
    }
    .h-item {
      background: var(--bg);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 10px 12px;
      font-size: 0.84rem;
    }
    .h-head {
      display: flex;
      justify-content: space-between;
      font-weight: 700;
      color: #6ee7b7;
    }
    .h-meta { color: var(--muted); font-size: 0.78rem; margin-top: 4px; }
    .error-msg { color: #f87171; font-size: 0.9rem; }
  </style>
</head>
<body>
  <div class="wrap">
    <header>
      <div>
        <h1>🌸 Sistema Web de Predicción de Flores</h1>
        <p>Ingresa medidas morfológicas y consulta el modelo vía <code>POST /predict</code></p>
      </div>
      <div class="links">
        <a href="/docs" target="_blank">Scalar Docs</a>
        <a href="/swagger" target="_blank">Swagger UI</a>
      </div>
    </header>

    <div class="grid">
      <section class="card">
        <h2>1. Parámetros de entrada</h2>
        <div class="sub">Medidas en centímetros o usa un ejemplo rápido:</div>

        <div class="presets">
          <button type="button" class="preset-btn" onclick="setVals(5.1, 3.5, 1.4, 0.2)">Ejemplo Setosa</button>
          <button type="button" class="preset-btn" onclick="setVals(6.0, 2.7, 4.5, 1.5)">Ejemplo Versicolor</button>
          <button type="button" class="preset-btn" onclick="setVals(6.9, 3.1, 5.4, 2.1)">Ejemplo Virginica</button>
        </div>

        <form id="form-flor">
          <div class="inputs">
            <div>
              <label for="sl">Largo del sépalo (cm)</label>
              <input id="sl" type="number" step="0.1" min="0.1" value="5.1" required />
            </div>
            <div>
              <label for="sw">Ancho del sépalo (cm)</label>
              <input id="sw" type="number" step="0.1" min="0.1" value="3.5" required />
            </div>
            <div>
              <label for="pl">Largo del pétalo (cm)</label>
              <input id="pl" type="number" step="0.1" min="0.1" value="1.4" required />
            </div>
            <div>
              <label for="pw">Ancho del pétalo (cm)</label>
              <input id="pw" type="number" step="0.1" min="0.1" value="0.2" required />
            </div>
          </div>

          <button type="submit" id="btn-enviar" class="btn-predict">Predecir especie de flor</button>
        </form>
      </section>

      <section class="card">
        <h2>2. Resultado de la predicción</h2>
        <div class="sub">Respuesta en tiempo real desde FastAPI</div>
        <div id="resultado">
          <div class="sub">Completa los parámetros y haz clic en <strong>Predecir especie de flor</strong>.</div>
        </div>
      </section>
    </div>

    <section class="card" style="margin-top: 20px;">
      <div class="history-top">
        <h2 style="margin:0;">3. Historial de consultas guardadas</h2>
        <button type="button" class="btn-clear" onclick="borrarHistorial()">Limpiar historial</button>
      </div>
      <div id="historial" class="history-list"></div>
    </section>
  </div>

  <script>
    const KEY_HISTORIAL = "historial_flores_fastapi_v1";

    function setVals(sl, sw, pl, pw) {
      document.getElementById("sl").value = sl;
      document.getElementById("sw").value = sw;
      document.getElementById("pl").value = pl;
      document.getElementById("pw").value = pw;
    }

    function leerHistorial() {
      try {
        return JSON.parse(localStorage.getItem(KEY_HISTORIAL) || "[]");
      } catch {
        return [];
      }
    }

    function guardarHistorial(data) {
      const actual = leerHistorial();
      actual.unshift({
        hora: new Date().toLocaleTimeString(),
        flor: data.flor,
        confianza: data.confianza_porcentaje,
        params: data.parametros
      });
      localStorage.setItem(KEY_HISTORIAL, JSON.stringify(actual.slice(0, 25)));
      pintarHistorial();
    }

    function borrarHistorial() {
      localStorage.removeItem(KEY_HISTORIAL);
      pintarHistorial();
    }

    function pintarHistorial() {
      const cont = document.getElementById("historial");
      const items = leerHistorial();
      if (!items.length) {
        cont.innerHTML = `<div class="sub">Sin consultas guardadas todavía.</div>`;
        return;
      }
      cont.innerHTML = items.map(it => `
        <div class="h-item">
          <div class="h-head">
            <span>🌺 ${it.flor} (${it.confianza}%)</span>
            <span>${it.hora}</span>
          </div>
          <div class="h-meta">
            Sépalo: ${it.params.sepal_length}×${it.params.sepal_width} cm ·
            Pétalo: ${it.params.petal_length}×${it.params.petal_width} cm
          </div>
        </div>
      `).join("");
    }

    function pintarResultado(data) {
      const probs = data.probabilidades || {};
      const barras = Object.entries(probs).map(([especie, pct]) => `
        <div class="prob-row">
          <div class="prob-label">
            <span>${especie}</span>
            <span>${pct}%</span>
          </div>
          <div class="bar-bg">
            <div class="bar-fill" style="width:${Math.min(100, Math.max(0, pct))}%"></div>
          </div>
        </div>
      `).join("");

      document.getElementById("resultado").innerHTML = `
        <div class="result-title">${data.flor}</div>
        <div class="desc">${data.descripcion || ""}</div>
        ${barras}
      `;
    }

    document.getElementById("form-flor").addEventListener("submit", async (e) => {
      e.preventDefault();
      const btn = document.getElementById("btn-enviar");
      const resBox = document.getElementById("resultado");

      const payload = {
        sepal_length: parseFloat(document.getElementById("sl").value),
        sepal_width: parseFloat(document.getElementById("sw").value),
        petal_length: parseFloat(document.getElementById("pl").value),
        petal_width: parseFloat(document.getElementById("pw").value)
      };

      btn.disabled = true;
      btn.textContent = "Prediciendo...";
      resBox.innerHTML = `<div class="sub">Consultando /predict en el servidor...</div>`;

      try {
        const resp = await fetch("/predict", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload)
        });
        const data = await resp.json();
        if (!resp.ok) {
          throw new Error(data.detail ? JSON.stringify(data.detail) : "Error en la predicción");
        }
        pintarResultado(data);
        guardarHistorial(data);
      } catch (err) {
        resBox.innerHTML = `<div class="error-msg">Error: ${err.message}</div>`;
      } finally {
        btn.disabled = false;
        btn.textContent = "Predecir especie de flor";
      }
    });

    pintarHistorial();
  </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def web_principal():
    return WEB_HTML
