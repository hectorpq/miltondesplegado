from fastapi import FastAPI, HTTPException, Path, Query
from fastapi.openapi.docs import get_swagger_ui_html
from scalar_fastapi import get_scalar_api_reference
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

# ─────────────────────────────────────────────
#  Configuración de la app
#  docs_url=None / redoc_url=None → desactiva
#  los endpoints por defecto para montarlos
#  manualmente con Scalar o Swagger custom
# ─────────────────────────────────────────────
app = FastAPI(
    title="Demo API",
    description="""
## API de demostración con FastAPI

Endpoints de ejemplo para gestión de **items**.

- Crear, leer, actualizar y eliminar items
- Documentación interactiva vía **Scalar** (`/docs`) o **Swagger UI** (`/swagger`)
""",
    version="1.0.0",
    docs_url=None,      # desactiva Swagger por defecto
    redoc_url=None,     # desactiva ReDoc por defecto
)


# ─────────────────────────────────────────────
#  Modelos Pydantic
# ─────────────────────────────────────────────
class ItemBase(BaseModel):
    nombre: str = Field(..., min_length=1, max_length=100, examples=["Laptop"])
    precio: float = Field(..., gt=0, examples=[1299.99])
    stock: int = Field(default=0, ge=0, examples=[10])
    descripcion: Optional[str] = Field(default=None, examples=["Laptop gamer 16GB RAM"])


class ItemCreate(ItemBase):
    pass


class ItemResponse(ItemBase):
    id: int
    creado_en: datetime

    model_config = {"from_attributes": True}


# ─────────────────────────────────────────────
#  "Base de datos" en memoria
# ─────────────────────────────────────────────
db: dict[int, dict] = {
    1: {"id": 1, "nombre": "Laptop",   "precio": 1299.99, "stock": 5,  "descripcion": "Laptop gamer",   "creado_en": datetime.now()},
    2: {"id": 2, "nombre": "Teclado",  "precio": 89.90,   "stock": 20, "descripcion": "Mecánico RGB",   "creado_en": datetime.now()},
    3: {"id": 3, "nombre": "Monitor",  "precio": 450.00,  "stock": 8,  "descripcion": '27" 144Hz',      "creado_en": datetime.now()},
}
next_id = 4


# ─────────────────────────────────────────────
#  Rutas de documentación (Scalar + Swagger)
# ─────────────────────────────────────────────
@app.get("/docs", include_in_schema=False)
async def scalar_docs():
    """Documentación interactiva con Scalar (moderna)."""
    return get_scalar_api_reference(
        openapi_url=app.openapi_url,
        title=app.title,
    )


@app.get("/swagger", include_in_schema=False)
async def swagger_docs():
    """Documentación interactiva con Swagger UI (clásica)."""
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=f"{app.title} – Swagger UI",
    )


# ─────────────────────────────────────────────
#  Endpoints
# ─────────────────────────────────────────────
@app.get(
    "/items",
    response_model=list[ItemResponse],
    summary="Listar items",
    tags=["Items"],
)
async def listar_items(
    skip: int = Query(default=0, ge=0, description="Registros a saltar"),
    limit: int = Query(default=10, ge=1, le=100, description="Máximo de registros"),
):
    """Retorna todos los items con paginación opcional."""
    items = list(db.values())
    return items[skip : skip + limit]


@app.get(
    "/items/{item_id}",
    response_model=ItemResponse,
    summary="Obtener item por ID",
    tags=["Items"],
)
async def obtener_item(
    item_id: int = Path(..., ge=1, description="ID del item"),
):
    """Busca un item por su ID. Lanza 404 si no existe."""
    if item_id not in db:
        raise HTTPException(status_code=404, detail=f"Item {item_id} no encontrado")
    return db[item_id]


@app.post(
    "/items",
    response_model=ItemResponse,
    status_code=201,
    summary="Crear item",
    tags=["Items"],
)
async def crear_item(item: ItemCreate):
    """Crea un nuevo item y lo persiste en memoria."""
    global next_id
    nuevo = {"id": next_id, "creado_en": datetime.now(), **item.model_dump()}
    db[next_id] = nuevo
    next_id += 1
    return nuevo


@app.put(
    "/items/{item_id}",
    response_model=ItemResponse,
    summary="Actualizar item",
    tags=["Items"],
)
async def actualizar_item(
    item_id: int = Path(..., ge=1),
    item: ItemCreate = ...,
):
    """Reemplaza completamente un item existente."""
    if item_id not in db:
        raise HTTPException(status_code=404, detail=f"Item {item_id} no encontrado")
    db[item_id].update(item.model_dump())
    return db[item_id]


@app.delete(
    "/items/{item_id}",
    status_code=204,
    summary="Eliminar item",
    tags=["Items"],
)
async def eliminar_item(
    item_id: int = Path(..., ge=1),
):
    """Elimina un item por ID. Retorna 204 sin cuerpo."""
    if item_id not in db:
        raise HTTPException(status_code=404, detail=f"Item {item_id} no encontrado")
    del db[item_id]


@app.get("/", include_in_schema=False)
async def root():
    return {
        "app": app.title,
        "version": app.version,
        "docs": "/docs (Scalar)",
        "swagger": "/swagger",
        "openapi": "/openapi.json",
    }
