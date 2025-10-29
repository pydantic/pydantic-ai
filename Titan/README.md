# 🚀 TITAN Agent Server - FastAPI + Pydantic AI + Telegram Bot

Una guía completa para principiantes: desde cero experiencia hasta tener tu servidor Python funcionando en producción.

## 📚 ¿Qué es esto?

Un servidor Python que:
1. Está corriendo 24/7 en internet
2. Recibe peticiones HTTP (como un webhook)
3. Ejecuta código Python (tu agente con Pydantic AI)
4. Devuelve respuestas vía Telegram

**Analogía**: Es como tener un asistente virtual que está siempre despierto esperando que le pidas algo.

## 🧰 Tecnologías Utilizadas

- **FastAPI**: Framework web para crear el servidor
- **Pydantic AI**: Biblioteca para trabajar con agentes de IA (con deferred tools)
- **Telegram Bot**: Para comunicarte con el agente
- **Supabase** (opcional): Para guardar datos
- **Railway/Fly.io**: Para deployment en la nube

## 📁 Estructura del Proyecto

```
Titan/
├── .env.example          # Plantilla de variables de entorno
├── .gitignore           # Archivos a ignorar en git
├── requirements.txt     # Dependencias de Python
├── Procfile            # Configuración para deployment
├── server.py           # Servidor FastAPI principal
├── agent.py            # Configuración del agente de IA
├── tools.py            # Tools del agente (ask_user)
└── README.md           # Esta guía
```

## 🛠️ Setup Local

### Paso 1: Verificar Python

```bash
python --version
# o
python3 --version
```

Necesitas Python 3.10 o superior.

#### Instalar Python (si no lo tienes)

**macOS:**
```bash
brew install python
```

**Windows:**
1. Ve a https://www.python.org/downloads/
2. Descarga el instalador
3. ¡IMPORTANTE! Marca "Add Python to PATH"
4. Instala

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install python3 python3-pip
```

### Paso 2: Crear Entorno Virtual

```bash
# Crear entorno virtual
python3 -m venv venv

# Activar entorno virtual
# macOS/Linux:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# Deberías ver (venv) al inicio de tu terminal
```

**¿Qué es un entorno virtual?** Es como una "burbuja" donde instalas cosas sin afectar tu sistema.

### Paso 3: Instalar Dependencias

```bash
pip install -r requirements.txt
```

Esto instalará:
- `fastapi`: Para crear el servidor
- `uvicorn`: Para correr el servidor
- `pydantic-ai`: Para el agente con deferred tools
- `python-telegram-bot`: Para comunicarte con Telegram
- `supabase`: Para guardar datos (opcional)
- `python-dotenv`: Para cargar variables de entorno

### Paso 4: Configurar Variables de Entorno

1. Copia `.env.example` a `.env`:
   ```bash
   cp .env.example .env
   ```

2. Edita `.env` con tus valores:
   ```bash
   ANTHROPIC_API_KEY=sk-ant-tu-api-key-aqui
   TELEGRAM_BOT_TOKEN=1234567890:ABCdefGHIjklMNOpqrsTUVwxyz
   PORT=8000
   ```

#### Obtener Token de Telegram

1. Abre Telegram
2. Busca `@BotFather`
3. Envía `/newbot`
4. Sigue las instrucciones
5. Copia el token que te da

#### Obtener API Key de Anthropic

1. Ve a https://console.anthropic.com/
2. Crea una cuenta o inicia sesión
3. Ve a "API Keys"
4. Crea una nueva key
5. Cópiala

### Paso 5: Correr el Servidor Localmente

```bash
python server.py
```

Deberías ver:
```
🚀 Iniciando servidor en puerto 8000...
📍 Webhook URL: http://localhost:8000/telegram/webhook

INFO:     Started server process
INFO:     Uvicorn running on http://0.0.0.0:8000
```

## 🧪 Testing Local con ngrok

Telegram necesita llamar a tu servidor, pero tu localhost no es accesible desde internet. Usamos ngrok:

### Instalar ngrok

**macOS:**
```bash
brew install ngrok
```

**Windows:**
Descarga de https://ngrok.com/download

**Linux:**
```bash
wget https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz
tar xvzf ngrok-v3-stable-linux-amd64.tgz
sudo mv ngrok /usr/local/bin
```

### Exponer tu servidor

```bash
ngrok http 8000
```

Deberías ver:
```
Forwarding   https://abc123.ngrok.io -> http://localhost:8000
```

Copia esa URL (https://abc123.ngrok.io).

### Configurar Webhook de Telegram

```bash
# Reemplaza:
# - <TU_TOKEN> con tu token de Telegram
# - <TU_NGROK_URL> con tu URL de ngrok

curl -X POST "https://api.telegram.org/bot<TU_TOKEN>/setWebhook?url=<TU_NGROK_URL>/telegram/webhook"
```

Respuesta esperada:
```json
{"ok":true,"result":true,"description":"Webhook was set"}
```

### Probar el Bot

1. Abre Telegram
2. Busca tu bot (el nombre que le pusiste)
3. Envía: "Quiero un plan de entrenamiento"

Deberías ver en tu terminal:
```
==================================================
[WEBHOOK] Mensaje de usuario 123456: Quiero un plan de entrenamiento
==================================================

[WEBHOOK] Iniciando agente para usuario 123456...
[AGENT] Ejecutando agente para usuario 123456...
[TOOL ask_user] Preguntando a usuario 123456: ¿Cuántos días a la semana puedes entrenar?
[TOOL ask_user] Esperando respuesta de usuario 123456...
```

En Telegram:
```
Bot: ¿Cuántos días a la semana puedes entrenar?
```

Responde:
```
Tu: 3 días
```

El bot seguirá haciendo preguntas hasta tener toda la información necesaria.

## 🚀 Deployment a Producción

### Opción 1: Railway (Recomendado)

#### Paso 1: Crear Cuenta
1. Ve a https://railway.app
2. Clic en "Start a New Project"
3. Login con GitHub

#### Paso 2: Subir Código a GitHub

```bash
# Inicializar git
git init
git add .
git commit -m "Initial commit"

# Crear repo en GitHub (en github.com):
# 1. Clic en "+" → "New repository"
# 2. Nombre: titan-agent-server
# 3. Clic en "Create repository"

# Conectar con GitHub (reemplaza <TU_USERNAME>)
git remote add origin https://github.com/<TU_USERNAME>/titan-agent-server.git
git branch -M main
git push -u origin main
```

#### Paso 3: Deploy en Railway

1. En Railway, clic en "New Project"
2. "Deploy from GitHub repo"
3. Selecciona `titan-agent-server`
4. Railway detectará automáticamente que es Python

#### Paso 4: Configurar Variables de Entorno

1. En Railway, clic en tu proyecto
2. Clic en "Variables"
3. Agregar:
   ```
   ANTHROPIC_API_KEY=tu-key
   TELEGRAM_BOT_TOKEN=tu-token
   PORT=8000
   ```

#### Paso 5: Obtener URL de Producción

1. Espera a que termine el deploy (~2 min)
2. Clic en "Settings" → "Generate Domain"
3. Copia tu URL (ej: `titan-agent-server-production.up.railway.app`)

#### Paso 6: Actualizar Webhook de Telegram

```bash
curl -X POST "https://api.telegram.org/bot<TU_TOKEN>/setWebhook?url=https://<TU_RAILWAY_URL>/telegram/webhook"
```

¡LISTO! Tu servidor está en producción 24/7.

### Opción 2: Fly.io

```bash
# Instalar flyctl
curl -L https://fly.io/install.sh | sh

# Login
flyctl auth login

# Crear app
flyctl launch

# Configurar secrets
flyctl secrets set ANTHROPIC_API_KEY=tu-key
flyctl secrets set TELEGRAM_BOT_TOKEN=tu-token

# Deploy
flyctl deploy
```

## 🔧 Debugging Común

### Problema 1: "ModuleNotFoundError: No module named 'pydantic_ai'"

**Solución:**
```bash
source venv/bin/activate  # Activa el entorno virtual
pip install -r requirements.txt
```

### Problema 2: El bot no responde

**Check 1: Verificar que el servidor está corriendo**
```bash
curl http://localhost:8000/health
# Debería devolver: {"status":"healthy"}
```

**Check 2: Verificar webhook configurado**
```bash
curl "https://api.telegram.org/bot<TU_TOKEN>/getWebhookInfo"
```

**Check 3: Ver logs en Railway**
1. Ve a Railway
2. Clic en tu proyecto
3. Clic en "Deployments" → Ver logs

### Problema 3: "Anthropic API key not found"

**Solución:** Asegúrate de que `.env` existe y tiene la key correcta.
```bash
cat .env  # Ver contenido del archivo
```

### Problema 4: El agente hace preguntas pero no espera respuesta

**Causa:** El webhook no está llegando a tu servidor.

**Solución:**
1. Verifica que el webhook de Telegram apunta a la URL correcta
2. Revisa los logs del servidor

## 📊 Cómo Funciona

### Flujo del Sistema

```
Usuario → [Telegram] → [Webhook] → [FastAPI Server]
                                         ↓
                                   [Agent (Pydantic AI)]
                                         ↓
                                   [Tool: ask_user]
                                         ↓
                                   [Telegram] → Usuario
```

### Componentes Principales

#### 1. `tools.py` - Deferred Tools
```python
async def ask_user(ctx: RunContext[dict], question: str) -> str:
    """
    Tool que pregunta al usuario vía Telegram y ESPERA la respuesta.
    """
    # 1. Envía pregunta por Telegram
    # 2. Crea un Future para esperar respuesta
    # 3. BLOQUEA hasta que el usuario responda
    # 4. Devuelve la respuesta al agente
```

**¿Qué es un Future?** Es una "promesa" de que habrá un valor en el futuro. Como cuando pides comida a domicilio: te dan un número de orden (Future) y esperas hasta que llegue la comida (valor).

#### 2. `agent.py` - Configuración del Agente
```python
agent = Agent(
    'claude-sonnet-4-20250514',  # Modelo de IA
    system_prompt=SYSTEM_PROMPT,  # Instrucciones
    tools=TOOLS,                  # ask_user
    retries=2                     # Reintentos si falla
)
```

#### 3. `server.py` - Servidor FastAPI

**Endpoint principal: `/telegram/webhook`**

Flujo:
1. Recibe mensaje de Telegram
2. **CASO 1**: Si hay respuesta pendiente → resuelve el Future
3. **CASO 2**: Si hay conversación activa → ignora (evita duplicados)
4. **CASO 3**: Nuevo mensaje → inicia agente en background

```python
# CASO 1: Resolver respuesta pendiente
if user_id in pending_responses:
    future = pending_responses.pop(user_id)
    future.set_result(text)  # ¡Cumple la promesa!

# CASO 3: Iniciar agente
asyncio.create_task(run_agent(user_id, text))
```

## 🎨 Personalización

### Cambiar el System Prompt

Edita `agent.py`:
```python
SYSTEM_PROMPT = """
Eres un [DESCRIPCIÓN DE TU AGENTE].

Tu trabajo es [OBJETIVO].

IMPORTANTE:
- [REGLA 1]
- [REGLA 2]
...
"""
```

### Agregar Más Tools

En `tools.py`:
```python
async def buscar_en_base_datos(ctx: RunContext[dict], query: str) -> str:
    """
    Tool que busca información en una base de datos.
    """
    # Tu lógica aquí
    return resultado

# Agregar a la lista
TOOLS = [
    ask_user,
    buscar_en_base_datos  # Nueva tool
]
```

### Integrar con n8n

En `server.py`, descomenta y configura `notify_n8n`:
```python
# Después de enviar respuesta final
await notify_n8n(user_id, result.data)
```

Agrega a `.env`:
```
N8N_WEBHOOK_URL=https://tu-n8n-webhook.com/...
```

## 💰 Costos

### Railway (Free Tier)
- $0/mes con límites:
  - 500 horas de ejecución/mes
  - 512 MB RAM
  - 1 GB disco

**Suficiente para**: 100-500 usuarios activos

### Cuando Escales (> 500 usuarios)
- Railway Pro: ~$5-20/mes
- O migrar a Fly.io: ~$5/mes

### Anthropic API
- Claude Sonnet: ~$3 por millón de tokens input, ~$15 por millón output
- **Estimado**: ~$0.01 por conversación
- 1000 conversaciones/mes ≈ $10

## 📚 Recursos Adicionales

- [Documentación de FastAPI](https://fastapi.tiangolo.com/)
- [Documentación de Pydantic AI](https://ai.pydantic.dev/)
- [Documentación de Telegram Bot API](https://core.telegram.org/bots/api)
- [Railway Docs](https://docs.railway.app/)

## ✅ Checklist de Setup

- [ ] Python instalado (3.10+)
- [ ] Entorno virtual creado
- [ ] Dependencias instaladas
- [ ] `.env` configurado con API keys
- [ ] Token de Telegram obtenido
- [ ] API key de Claude obtenida
- [ ] Servidor corriendo localmente
- [ ] Ngrok funcionando (para testing local)
- [ ] Webhook de Telegram configurado
- [ ] Prueba local exitosa
- [ ] Código subido a GitHub
- [ ] Deploy en Railway/Fly.io exitoso
- [ ] Variables de entorno en producción configuradas
- [ ] Webhook actualizado a URL de producción
- [ ] Prueba en producción exitosa

## 🆘 Soporte

Si te atascas en algún paso:

1. **Revisa los logs**: Terminal local o Railway
2. **Verifica variables de entorno**: `.env` local o Railway
3. **Comprueba el webhook**: `getWebhookInfo`
4. **Lee la sección de Debugging**

## 📝 Licencia

Este proyecto es de código abierto. Siéntete libre de usarlo y modificarlo como quieras.

---

¡Hecho con ❤️ para ayudarte a crear tu primer agente de IA con Python!
