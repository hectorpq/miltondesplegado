from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from scalar_fastapi import get_scalar_api_reference
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

app = FastAPI(
    title="API de Predicción de Flores",
    version="2.0.0",
    docs_url="/swagger",
    redoc_url=None,
)

# Entrenamiento rápido al iniciar la app con el dataset Iris
iris = load_iris()
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(iris.data, iris.target)

# Nombres amigables en español
SPECIES_MAP = {
    0: "Iris Setosa",
    1: "Iris Versicolor",
    2: "Iris Virginica",
}


class FlorInput(BaseModel):
    sepal_length: float = Field(..., gt=0, description="Largo del sépalo en cm (ej. 5.1)")
    sepal_width: float = Field(..., gt=0, description="Ancho del sépalo en cm (ej. 3.5)")
    petal_length: float = Field(..., gt=0, description="Largo del pétalo en cm (ej. 1.4)")
    petal_width: float = Field(..., gt=0, description="Ancho del pétalo en cm (ej. 0.2)")


@app.get("/")
def root():
    return {
        "app": "API de Predicción de Flores",
        "version": "2.0.0",
        "predict_endpoint": "/predict",
        "web": "/web",
        "docs": "/docs",
        "swagger": "/swagger",
    }


@app.post("/predict")
def predecir_flor(flor: FlorInput):
    features = [[
        flor.sepal_length,
        flor.sepal_width,
        flor.petal_length,
        flor.petal_width,
    ]]
    pred_index = int(model.predict(features)[0])
    probs = model.predict_proba(features)[0]

    probabilidades = {
        SPECIES_MAP[i]: round(float(prob), 4)
        for i, prob in enumerate(probs)
    }

    return {
        "flor_predicha": SPECIES_MAP[pred_index],
        "clase_id": pred_index,
        "probabilidades": probabilidades,
        "parametros_recibidos": flor.model_dump(),
    }


@app.get("/docs", include_in_schema=False)
def scalar_docs():
    return get_scalar_api_reference(
        openapi_url=app.openapi_url,
        title=app.title + " - Scalar",
    )


@app.get("/web", response_class=HTMLResponse, include_in_schema=False)
def interfaz_web():
    return """
    <!DOCTYPE html>
    <html lang="es">
    <head>
      <meta charset="UTF-8" />
      <meta name="viewport" content="width=device-width, initial-scale=1.0" />
      <title>Predicción de Flores</title>
      <style>
        body { font-family: Arial, sans-serif; max-width: 520px; margin: 40px auto; padding: 0 16px; background: #111; color: #eee; }
        .card { background: #1c1c1c; padding: 20px; border-radius: 10px; border: 1px solid #333; }
        label { display: block; margin-top: 10px; font-size: 14px; }
        input { width: 100%; padding: 8px; margin-top: 4px; border-radius: 6px; border: 1px solid #444; background: #111; color: #fff; box-sizing: border-box; }
        button { margin-top: 16px; width: 100%; padding: 10px; border: none; border-radius: 6px; background: #2563eb; color: white; font-weight: bold; cursor: pointer; }
        button:hover { background: #1d4ed8; }
        pre { background: #0d0d0d; padding: 12px; border-radius: 8px; overflow-x: auto; margin-top: 16px; }
      </style>
    </head>
    <body>
      <div class="card">
        <h2>🌸 Predictor de Tipo de Flor</h2>
        <p>Ingresa las medidas de la flor para predecir su especie:</p>
        <label>Largo del sépalo (cm): <input id="sl" type="number" step="0.1" value="5.1" /></label>
        <label>Ancho del sépalo (cm): <input id="sw" type="number" step="0.1" value="3.5" /></label>
        <label>Largo del pétalo (cm): <input id="pl" type="number" step="0.1" value="1.4" /></label>
        <label>Ancho del pétalo (cm): <input id="pw" type="number" step="0.1" value="0.2" /></label>
        <button onclick="predecir()">Predecir flor</button>
        <pre id="resultado">Esperando datos...</pre>
      </div>
      <script>
        async function predecir() {
          const payload = {
            sepal_length: parseFloat(document.getElementById("sl").value),
            sepal_width: parseFloat(document.getElementById("sw").value),
            petal_length: parseFloat(document.getElementById("pl").value),
            petal_width: parseFloat(document.getElementById("pw").value)
          };
          const res = await fetch("/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
          });
          const data = await res.json();
          document.getElementById("resultado").textContent = JSON.stringify(data, null, 2);
        }
      </script>
    </body>
    </html>
    """
