# Deployment Explicado

## ¿Qué es Deployment?

**Deployment** = Subir tu código a un servidor en la nube para que esté disponible 24/7.

### Analogía: Tienda Física vs Online

**Sin deployment** (localhost):
```
Tu código corre en tu laptop
↓
Solo tú puedes acceder
↓
Si apagas tu laptop, el bot muere
↓
No es accesible desde internet
```

Como tener una tienda en tu garage: solo tú puedes entrar.

**Con deployment** (producción):
```
Tu código corre en un servidor en la nube
↓
Cualquiera puede acceder (con el link)
↓
Está corriendo 24/7
↓
Accesible desde internet
```

Como tener una tienda en una calle céntrica: cualquiera puede entrar.

## ¿Por qué Necesitamos Deployment?

### Problema 1: Laptop Apagada

```
[Tu laptop]
    ↓
[Tu bot corriendo]
    ↓
[Apagas la laptop]
    ↓
[Bot muerto] ☠️
```

### Problema 2: No es accesible

```
[Telegram]
    ↓
Intenta llamar a http://localhost:8000
    ↓
❌ Error: No puede acceder
```

Telegram está en internet, tu localhost no.

### Solución: Servidor en la Nube

```
[Servidor en la nube]
    ↓
[Corriendo 24/7]
    ↓
[Accesible desde internet]
    ↓
[Telegram puede llamar al webhook]
```

## Opciones de Deployment

### 1. Railway (Recomendado para principiantes)

**Ventajas**:
- ✅ Muy fácil
- ✅ Deploy con un clic
- ✅ Free tier generoso
- ✅ HTTPS automático
- ✅ Logs fáciles de ver

**Desventajas**:
- ❌ Más caro al escalar

**Ideal para**: Principiantes, prototipos, proyectos pequeños

### 2. Fly.io

**Ventajas**:
- ✅ Más barato
- ✅ Control fino
- ✅ Global (múltiples regiones)

**Desventajas**:
- ❌ Más complejo (requiere CLI)
- ❌ Configuración manual

**Ideal para**: Usuarios con experiencia, producción seria

### 3. Heroku

**Ventajas**:
- ✅ Muy popular
- ✅ Fácil de usar

**Desventajas**:
- ❌ Ya no tiene free tier
- ❌ Más caro

**Ideal para**: Proyectos con presupuesto

### 4. VPS (DigitalOcean, Linode, AWS EC2)

**Ventajas**:
- ✅ Control total
- ✅ Barato al escalar

**Desventajas**:
- ❌ Muy complejo
- ❌ Tienes que configurar todo
- ❌ No recomendado para principiantes

**Ideal para**: Expertos, proyectos grandes

## Deployment con Railway (Paso a Paso Detallado)

### Paso 1: Preparar el Código

#### 1.1. Verificar estructura

```
Titan/
├── server.py
├── agent.py
├── tools.py
├── requirements.txt
├── Procfile
├── .gitignore
└── .env.example
```

#### 1.2. Verificar requirements.txt

```txt
fastapi==0.104.1
uvicorn==0.24.0
pydantic-ai==0.0.14
python-telegram-bot==20.7
supabase==2.3.0
python-dotenv==1.0.0
httpx==0.25.2
```

#### 1.3. Verificar Procfile

```
web: python server.py
```

**¿Qué es Procfile?**
- Archivo que dice cómo correr tu app
- Railway lo lee y ejecuta el comando

#### 1.4. Verificar .gitignore

```
venv/
__pycache__/
*.pyc
.env
.DS_Store
```

**¿Por qué?**
- No queremos subir archivos innecesarios
- `.env` tiene secretos (NUNCA subirlo a git)

### Paso 2: Subir a GitHub

#### 2.1. Inicializar git (si no lo has hecho)

```bash
cd Titan
git init
```

#### 2.2. Agregar archivos

```bash
git add .
```

**¿Qué hace?**
- Prepara todos los archivos para commit
- Ignora archivos en `.gitignore`

#### 2.3. Commit

```bash
git commit -m "Initial commit: TITAN agent server"
```

**¿Qué es un commit?**
- Guarda un snapshot de tu código
- Como tomar una foto en el tiempo

#### 2.4. Crear repo en GitHub

1. Ve a https://github.com
2. Clic en "+" (arriba derecha) → "New repository"
3. Nombre: `titan-agent-server`
4. **NO** marques "Initialize with README"
5. Clic en "Create repository"

#### 2.5. Conectar con GitHub

```bash
# Reemplaza <TU_USERNAME> con tu usuario de GitHub
git remote add origin https://github.com/<TU_USERNAME>/titan-agent-server.git
git branch -M main
git push -u origin main
```

**¿Qué hace cada comando?**
- `git remote add origin`: Conecta tu repo local con GitHub
- `git branch -M main`: Renombra rama a "main"
- `git push -u origin main`: Sube tu código a GitHub

### Paso 3: Crear Cuenta en Railway

1. Ve a https://railway.app
2. Clic en "Login" (arriba derecha)
3. Selecciona "Login with GitHub"
4. Autoriza Railway

**¿Por qué GitHub?**
- Railway necesita acceso a tus repos
- Deploy automático cuando haces push

### Paso 4: Crear Proyecto en Railway

#### 4.1. Nuevo proyecto

1. En Railway dashboard, clic en "New Project"
2. Selecciona "Deploy from GitHub repo"

#### 4.2. Seleccionar repo

1. Busca `titan-agent-server`
2. Clic en el repo

**¿Qué pasa ahora?**
- Railway clona tu repo
- Detecta que es Python (ve `requirements.txt`)
- Instala dependencias automáticamente
- Intenta correr (fallará porque faltan variables de entorno)

#### 4.3. Ver logs

1. Clic en tu proyecto
2. Clic en "Deployments"
3. Clic en el deployment activo
4. Ver logs en tiempo real

Verás algo como:
```
Installing dependencies from requirements.txt...
Running: python server.py
ValueError: Faltan variables de entorno
```

Es normal, necesitamos configurar variables de entorno.

### Paso 5: Configurar Variables de Entorno

#### 5.1. Ir a Variables

1. En tu proyecto, clic en "Variables" (tab arriba)
2. Verás una lista vacía

#### 5.2. Agregar variables

Clic en "New Variable" y agrega:

```
ANTHROPIC_API_KEY=sk-ant-tu-api-key-aqui
TELEGRAM_BOT_TOKEN=1234567890:ABCdefGHIjklMNOpqrsTUVwxyz
PORT=8000
```

**IMPORTANTE**: Usa tus valores reales.

#### 5.3. Redeploy automático

Cuando guardas variables, Railway automáticamente:
1. Detiene el deployment actual
2. Crea uno nuevo con las variables
3. Corre tu código

Espera 1-2 minutos.

### Paso 6: Obtener URL Pública

#### 6.1. Generar dominio

1. En tu proyecto, clic en "Settings"
2. Scroll hasta "Domains"
3. Clic en "Generate Domain"

Railway te da una URL como:
```
titan-agent-server-production-a1b2.up.railway.app
```

#### 6.2. Verificar que funciona

Abre en navegador:
```
https://titan-agent-server-production-a1b2.up.railway.app/health
```

Deberías ver:
```json
{"status":"healthy"}
```

¡Funciona! 🎉

### Paso 7: Configurar Webhook de Telegram

Ahora que tu servidor está en internet, configura el webhook:

```bash
curl -X POST "https://api.telegram.org/bot<TU_TOKEN>/setWebhook?url=https://<TU_RAILWAY_URL>/telegram/webhook"
```

Ejemplo:
```bash
curl -X POST "https://api.telegram.org/bot1234567890:ABC.../setWebhook?url=https://titan-agent-server-production-a1b2.up.railway.app/telegram/webhook"
```

Respuesta:
```json
{"ok":true,"result":true,"description":"Webhook was set"}
```

### Paso 8: Probar en Producción

1. Abre Telegram
2. Busca tu bot
3. Envía: "Quiero un plan de entrenamiento"

**Ver logs en Railway**:
1. Ve a Railway
2. Tu proyecto → "Deployments"
3. Clic en deployment activo
4. Ver logs en tiempo real

Deberías ver:
```
==================================================
[WEBHOOK] Mensaje de usuario 123456: Quiero un plan de entrenamiento
==================================================
[AGENT] Ejecutando agente para usuario 123456...
[TOOL ask_user] Preguntando a usuario 123456: ¿Cuántos días...
```

¡Funcionando en producción! 🚀

## ¿Qué Pasa Detrás de Escena?

### Cuando haces push a GitHub

```
[Tu laptop]
    ↓
git push
    ↓
[GitHub recibe código]
    ↓
[GitHub notifica a Railway] (webhook)
    ↓
[Railway se entera de cambios]
```

### Railway Deploy Process

```
[Railway clona repo]
    ↓
[Detecta Python] (ve requirements.txt)
    ↓
[Crea contenedor Docker]
    ↓
[Instala Python]
    ↓
[Instala dependencias] (pip install -r requirements.txt)
    ↓
[Corre Procfile] (python server.py)
    ↓
[Expone puerto] (el que especificaste en PORT)
    ↓
[Asigna URL pública]
    ↓
[Deployment exitoso] ✅
```

## Auto-Deploy

Railway hace **auto-deploy**: cada vez que haces push a GitHub, redeploys automáticamente.

### Ejemplo

```bash
# Haces un cambio en agent.py
# Guardas el archivo

git add agent.py
git commit -m "Mejora system prompt"
git push

# Railway automáticamente:
# 1. Detecta el push
# 2. Clona el nuevo código
# 3. Redeploys
# 4. En 1-2 minutos, el cambio está en producción
```

## Logs y Debugging

### Ver Logs en Tiempo Real

1. Railway → Tu proyecto → "Deployments"
2. Clic en deployment activo
3. Ver logs

**Tipos de logs**:
- **Build logs**: Instalación de dependencias
- **Runtime logs**: Tu código corriendo (tus prints)

### Ejemplo de logs

```
[BUILD]
Installing fastapi==0.104.1
Installing uvicorn==0.24.0
...
Build successful!

[RUNTIME]
🚀 Iniciando servidor en puerto 8000...
INFO:     Started server process
INFO:     Uvicorn running on http://0.0.0.0:8000

[WEBHOOK] Mensaje de usuario 123456: Hola
[AGENT] Ejecutando agente...
```

### Debugging Errores

#### Error: "Port already in use"

**Causa**: Railway asigna el puerto automáticamente.

**Solución**: Usar variable de entorno:
```python
port = int(os.getenv("PORT", 8000))
```

#### Error: "Module not found"

**Causa**: Falta en `requirements.txt`.

**Solución**: Agregar a `requirements.txt`:
```bash
pip freeze > requirements.txt
git add requirements.txt
git commit -m "Fix dependencies"
git push
```

#### Error: Webhook no llega

**Causa**: URL incorrecta o servidor caído.

**Verificar**:
```bash
# 1. Ver si el servidor responde
curl https://tu-url.up.railway.app/health

# 2. Ver webhook configurado
curl "https://api.telegram.org/bot<TOKEN>/getWebhookInfo"
```

## Costos

### Railway Free Tier

- **$0/mes**
- **$5 de crédito gratis**
- **500 horas de ejecución/mes**
- **512 MB RAM**
- **1 GB disco**

**¿Cuánto dura?**
- 500 horas = 20.8 días de 24/7
- Suficiente para: 100-500 usuarios activos/mes

### Cuándo Pagarás

Cuando superes el free tier:
- **$5/mes** por 100 horas adicionales
- **$20/mes** para uso ilimitado

**Estimación para 1000 usuarios/mes**: ~$5-10/mes

## Deployment con Fly.io (Alternativa)

### Ventajas

- Más barato
- Más control

### Pasos Básicos

```bash
# 1. Instalar flyctl
curl -L https://fly.io/install.sh | sh

# 2. Login
flyctl auth login

# 3. Crear app
flyctl launch

# 4. Configurar secrets
flyctl secrets set ANTHROPIC_API_KEY=sk-ant-...
flyctl secrets set TELEGRAM_BOT_TOKEN=123:ABC...

# 5. Deploy
flyctl deploy
```

## Checklist de Deployment

- [ ] Código funciona localmente
- [ ] `requirements.txt` tiene todas las dependencias
- [ ] `Procfile` configurado correctamente
- [ ] `.gitignore` incluye `.env` y `venv/`
- [ ] Variables de entorno documentadas en `.env.example`
- [ ] Código subido a GitHub
- [ ] Proyecto creado en Railway
- [ ] Variables de entorno configuradas en Railway
- [ ] Dominio generado
- [ ] Webhook de Telegram configurado con URL de producción
- [ ] Prueba en producción exitosa

## Tips para Producción

### 1. Logs Útiles

Agrega logs en lugares clave:

```python
print(f"[INFO] Usuario {user_id} envió mensaje")
print(f"[DEBUG] Deps: {ctx.deps}")
print(f"[ERROR] Error: {e}")
```

### 2. Error Handling

Siempre maneja errores:

```python
try:
    result = await agent.run(...)
except Exception as e:
    print(f"[ERROR] Error ejecutando agente: {e}")
    await bot.send_message(user_id, "Error, intenta de nuevo")
```

### 3. Health Check

Útil para monitoreo:

```python
@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }
```

### 4. Versioning

Agrega versión a tu API:

```python
@app.get("/")
async def root():
    return {
        "status": "ok",
        "version": "1.0.0",
        "message": "TITAN Agent Server"
    }
```

## Resumen

- **Deployment**: Subir código a servidor en la nube
- **Railway**: Plataforma fácil para deployment
- **Proceso**: GitHub → Railway → URL pública
- **Auto-deploy**: Push a GitHub → redeploy automático
- **Variables de entorno**: Configurar en Railway, no en código
- **Logs**: Ver en tiempo real en Railway dashboard
- **Costos**: Free tier suficiente para empezar

Con deployment, tu bot pasa de un proyecto local a un servicio profesional disponible 24/7.
