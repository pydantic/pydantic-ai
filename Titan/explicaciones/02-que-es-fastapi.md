# ¿Qué es FastAPI?

## Analogía Simple

Imagina que quieres abrir un restaurante, pero no quieres construir el edificio desde cero.

**Sin FastAPI**: Tendrías que:
- Construir las paredes
- Instalar la cocina
- Poner las mesas
- Contratar personal
- Crear el sistema de pedidos
- Todo desde cero...

**Con FastAPI**: Te dan un restaurante pre-armado:
- Ya tiene cocina equipada
- Sistema de pedidos funcionando
- Personal básico
- Solo necesitas: agregar tu menú (tu lógica)

## ¿Qué es FastAPI?

FastAPI es un **framework web** de Python. Un framework es un conjunto de herramientas que te facilita crear aplicaciones web.

### Sin FastAPI (código crudo):
```python
# Tendrías que escribir MUCHO código para:
import socket
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(('0.0.0.0', 8000))
server.listen()
# ... 200 líneas más de código bajo nivel ...
```

### Con FastAPI (código simple):
```python
from fastapi import FastAPI
app = FastAPI()

@app.get("/")
def root():
    return {"message": "Hola"}
```

¡Solo 5 líneas!

## Componentes de FastAPI en Nuestro Proyecto

### 1. Crear la App

```python
app = FastAPI(title="TITAN Agent Server")
```

Esto crea tu servidor. Es como decir: "Quiero un restaurante llamado TITAN Agent Server".

### 2. Definir Endpoints (Rutas)

```python
@app.get("/")
async def root():
    return {"status": "ok"}
```

**Desglose**:
- `@app.get("/")`: Decorador que dice "cuando alguien haga GET a `/`, ejecuta esta función"
- `async def root()`: Función asíncrona (puede esperar sin bloquear)
- `return {...}`: Lo que se devuelve al cliente

### 3. Recibir Datos

```python
@app.post("/telegram/webhook")
async def telegram_webhook(request: Request):
    data = await request.json()  # Obtener datos del body
    # ... procesar datos ...
    return {"ok": True}
```

**¿Qué pasa aquí?**
1. Alguien envía POST a `/telegram/webhook`
2. FastAPI recibe la petición
3. Extraemos los datos con `request.json()`
4. Procesamos
5. Devolvemos respuesta

## Ventajas de FastAPI

### 1. Automático y Rápido
- **Validación**: Verifica que los datos sean correctos
- **Documentación**: Crea docs automáticamente
- **Serialización**: Convierte Python a JSON automáticamente

### 2. Type Hints
```python
def sumar(a: int, b: int) -> int:
    return a + b
```

FastAPI usa estos "hints" para:
- Validar que `a` y `b` sean enteros
- Convertir automáticamente strings a enteros si es posible
- Documentar los tipos esperados

### 3. Async/Await
```python
async def hacer_algo():
    await operacion_lenta()  # No bloquea, permite que otras cosas corran
```

Esto permite que tu servidor maneje múltiples peticiones al mismo tiempo.

## Comparación con Otras Tecnologías

### FastAPI vs Flask

**Flask** (más viejo):
```python
from flask import Flask
app = Flask(__name__)

@app.route("/")
def root():
    return {"message": "Hola"}
```

**FastAPI** (más moderno):
```python
from fastapi import FastAPI
app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hola"}
```

Diferencias:
- FastAPI tiene async/await nativo
- FastAPI genera documentación automática
- FastAPI valida tipos automáticamente
- FastAPI es más rápido

### FastAPI vs Node.js (Express)

**Node.js**:
```javascript
const express = require('express')
const app = express()

app.get('/', (req, res) => {
  res.json({message: 'Hola'})
})
```

**FastAPI**:
```python
from fastapi import FastAPI
app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hola"}
```

Ambos son buenos. FastAPI es ideal si ya sabes Python.

## Request y Response

### Request (Petición)

Cuando alguien hace una petición a tu servidor:

```python
@app.post("/telegram/webhook")
async def webhook(request: Request):
    # Request contiene:
    data = await request.json()  # Body (datos enviados)
    headers = request.headers     # Headers (metadatos)
    method = request.method       # Método (GET, POST, etc.)
```

### Response (Respuesta)

Lo que tu servidor devuelve:

```python
return {"ok": True, "message": "Success"}  # FastAPI convierte a JSON automáticamente
```

## Decoradores (@app.get, @app.post)

Un **decorador** es como una etiqueta que modifica una función.

```python
@app.get("/health")
async def health():
    return {"status": "healthy"}
```

**Sin decorador** (no funcionaría):
```python
async def health():
    return {"status": "healthy"}
```

El servidor no sabría cuándo llamar a esta función.

**Con decorador**:
FastAPI dice: "Ah, cuando reciba GET a `/health`, llamaré a esta función".

## Path Parameters

Puedes tener URLs dinámicas:

```python
@app.get("/users/{user_id}")
async def get_user(user_id: int):
    return {"user_id": user_id, "name": f"User {user_id}"}
```

Si alguien accede a:
- `/users/123` → `user_id = 123`
- `/users/456` → `user_id = 456`

## Query Parameters

```python
@app.get("/search")
async def search(q: str, limit: int = 10):
    return {"query": q, "limit": limit}
```

URLs:
- `/search?q=hola` → `q = "hola"`, `limit = 10` (default)
- `/search?q=hola&limit=20` → `q = "hola"`, `limit = 20`

## Body Parameters

```python
from pydantic import BaseModel

class Message(BaseModel):
    text: str
    user_id: int

@app.post("/send")
async def send_message(message: Message):
    return {"received": message.text}
```

Cliente envía:
```json
POST /send
{
  "text": "Hola",
  "user_id": 123
}
```

FastAPI automáticamente:
1. Parsea el JSON
2. Valida que `text` sea string
3. Valida que `user_id` sea int
4. Crea un objeto `Message`

## Documentación Automática

FastAPI genera docs gratis. Corre tu servidor y ve a:

- `http://localhost:8000/docs` → Swagger UI (interactivo)
- `http://localhost:8000/redoc` → ReDoc (bonito)

Puedes **probar tus endpoints** directamente desde el navegador.

## Middleware

Código que se ejecuta **antes** o **después** de cada request:

```python
@app.middleware("http")
async def log_requests(request: Request, call_next):
    print(f"Request: {request.url}")  # ANTES
    response = await call_next(request)
    print(f"Response: {response.status_code}")  # DESPUÉS
    return response
```

## Exception Handling

Manejo de errores:

```python
from fastapi import HTTPException

@app.get("/users/{user_id}")
async def get_user(user_id: int):
    if user_id not in database:
        raise HTTPException(status_code=404, detail="User not found")
    return {"user_id": user_id}
```

Si el usuario no existe, devuelve:
```json
{
  "detail": "User not found"
}
```
Con status code 404.

## Nuestro server.py Explicado

```python
from fastapi import FastAPI, Request

# Crear app
app = FastAPI(title="TITAN Agent Server")

# Endpoint raíz
@app.get("/")
async def root():
    return {"status": "ok", "message": "TITAN Agent Server is running"}

# Health check
@app.get("/health")
async def health():
    return {"status": "healthy"}

# Webhook de Telegram
@app.post("/telegram/webhook")
async def telegram_webhook(request: Request):
    # Recibir datos de Telegram
    data = await request.json()

    # Procesar mensaje
    # ...

    return {"ok": True}
```

## Correr el Servidor

Con **Uvicorn** (servidor ASGI):

```python
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

O desde terminal:
```bash
uvicorn server:app --reload
```

**Parámetros**:
- `server`: Archivo `server.py`
- `app`: Variable del FastAPI app
- `--reload`: Reinicia automáticamente cuando cambias código (solo desarrollo)

## Resumen

- **FastAPI**: Framework para crear servidores web en Python
- **Decoradores** (`@app.get`, `@app.post`): Definen endpoints
- **Request**: Datos que llegan al servidor
- **Response**: Datos que el servidor devuelve
- **Async/Await**: Permite manejar múltiples peticiones simultáneamente
- **Documentación automática**: `/docs` y `/redoc`

FastAPI hace que crear un servidor sea tan fácil como escribir funciones de Python normales.
