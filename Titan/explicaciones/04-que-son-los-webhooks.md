# ¿Qué son los Webhooks?

## Analogía Simple

### Sin Webhook (Polling): El Cartero Impaciente

Imagina que estás esperando un paquete:

**Polling** (preguntando constantemente):
```
Tú: ¿Ya llegó mi paquete?
Correo: No.
[Esperas 1 minuto]
Tú: ¿Ya llegó mi paquete?
Correo: No.
[Esperas 1 minuto]
Tú: ¿Ya llegó mi paquete?
Correo: No.
[Esperas 1 minuto]
Tú: ¿Ya llegó mi paquete?
Correo: Sí, aquí está.
```

**Problemas**:
- Desperdicias tiempo preguntando
- El correo se molesta con tantas preguntas
- Ineficiente

### Con Webhook: El Timbre

**Webhook** (te avisan cuando algo sucede):
```
[Trabajas tranquilamente]
[Suena el timbre]
Tú: ¡Ah! Llegó mi paquete.
```

**Ventajas**:
- Solo te interrumpen cuando es necesario
- No desperdicias recursos
- Eficiente

## ¿Qué es un Webhook?

Un **webhook** es una forma de que un servicio te **notifique** cuando algo sucede, en lugar de que tú estés preguntando constantemente.

**Definición técnica**: Un webhook es una URL de tu servidor que otro servicio llama (POST request) cuando ocurre un evento.

## Ejemplo con Telegram

### Sin Webhook (getUpdates - Polling)

```python
# Tu código pregunta constantemente
while True:
    updates = telegram.getUpdates()  # "¿Hay mensajes nuevos?"
    if updates:
        procesar_mensaje(updates)
    time.sleep(1)  # Espera 1 segundo
```

**Problemas**:
- Tu servidor está constantemente haciendo requests
- Desperdicia ancho de banda
- Latencia: puede haber delay de hasta 1 segundo

### Con Webhook

```python
# Telegram te llama cuando hay un mensaje nuevo
@app.post("/telegram/webhook")
async def telegram_webhook(request: Request):
    data = await request.json()  # Telegram te envía el mensaje
    procesar_mensaje(data)
    return {"ok": True}
```

**Ventajas**:
- Telegram te notifica inmediatamente
- Tu servidor no hace requests innecesarios
- Más eficiente

## Cómo Funciona un Webhook

### 1. Configuración (Una sola vez)

Tú le dices a Telegram: "Cuando haya un mensaje nuevo, llama a esta URL":

```bash
curl -X POST "https://api.telegram.org/bot<TOKEN>/setWebhook?url=https://tu-servidor.com/telegram/webhook"
```

Telegram responde:
```json
{"ok": true, "result": true, "description": "Webhook was set"}
```

### 2. Funcionamiento Normal

```
Usuario → Envía mensaje "Hola" a tu bot en Telegram
    ↓
Telegram → Recibe el mensaje
    ↓
Telegram → Hace POST a https://tu-servidor.com/telegram/webhook
    ↓
{
  "message": {
    "text": "Hola",
    "from": {"id": 123456}
  }
}
    ↓
Tu servidor → Recibe el POST
    ↓
Tu servidor → Procesa el mensaje
    ↓
Tu servidor → Responde {"ok": true}
    ↓
[Opcionalmente] Tu servidor → Envía respuesta al usuario vía Telegram API
```

## Anatomía de un Webhook

### Endpoint del Webhook

```python
@app.post("/telegram/webhook")  # La URL que Telegram llamará
async def telegram_webhook(request: Request):
    # ...
```

**URL completa**: `https://tu-servidor.com/telegram/webhook`

### Recibir Datos

```python
async def telegram_webhook(request: Request):
    # Obtener datos que Telegram envió
    data = await request.json()

    # data contiene:
    # {
    #   "update_id": 123456789,
    #   "message": {
    #     "message_id": 1,
    #     "from": {"id": 123456, "first_name": "Juan"},
    #     "text": "Hola"
    #   }
    # }
```

### Parsear el Update

```python
from telegram import Update, Bot

async def telegram_webhook(request: Request):
    data = await request.json()

    # Convertir a objeto Update de python-telegram-bot
    update = Update.de_json(data, bot)

    # Acceder a los datos
    if update.message:
        user_id = update.message.from_user.id
        text = update.message.text
        print(f"Usuario {user_id} dijo: {text}")
```

### Responder

```python
async def telegram_webhook(request: Request):
    # ... procesar mensaje ...

    # IMPORTANTE: Debes devolver una respuesta rápido
    return {"ok": True}
```

**¿Por qué rápido?**
- Telegram espera respuesta en menos de 60 segundos
- Si tarda mucho, Telegram reintenta
- Puedes procesar en background (asyncio.create_task)

## Webhook en Nuestro Proyecto

### Configurar el Webhook

```bash
# Reemplaza:
# - <TU_TOKEN> con tu token de Telegram
# - <TU_URL> con tu URL de servidor

curl -X POST "https://api.telegram.org/bot<TU_TOKEN>/setWebhook?url=<TU_URL>/telegram/webhook"
```

**Localmente (con ngrok)**:
```bash
curl -X POST "https://api.telegram.org/bot<TU_TOKEN>/setWebhook?url=https://abc123.ngrok.io/telegram/webhook"
```

**En producción (Railway)**:
```bash
curl -X POST "https://api.telegram.org/bot<TU_TOKEN>/setWebhook?url=https://tu-proyecto.up.railway.app/telegram/webhook"
```

### Verificar el Webhook

```bash
curl "https://api.telegram.org/bot<TU_TOKEN>/getWebhookInfo"
```

Respuesta:
```json
{
  "ok": true,
  "result": {
    "url": "https://tu-servidor.com/telegram/webhook",
    "has_custom_certificate": false,
    "pending_update_count": 0,
    "last_error_date": 0,
    "max_connections": 40
  }
}
```

**Campos importantes**:
- `url`: La URL configurada
- `pending_update_count`: Mensajes pendientes (debería ser 0)
- `last_error_date`: Si hay errores, cuándo fue el último

### Eliminar el Webhook

```bash
curl -X POST "https://api.telegram.org/bot<TU_TOKEN>/deleteWebhook"
```

Útil si quieres volver a polling o cambiar de URL.

## Flujo Completo en Nuestro Proyecto

### 1. Usuario envía mensaje

```
Usuario escribe: "Quiero un plan de entrenamiento"
```

### 2. Telegram notifica tu servidor

```http
POST https://tu-servidor.com/telegram/webhook
Content-Type: application/json

{
  "update_id": 123456789,
  "message": {
    "message_id": 1,
    "from": {
      "id": 123456,
      "first_name": "Juan"
    },
    "text": "Quiero un plan de entrenamiento"
  }
}
```

### 3. Tu servidor recibe y procesa

```python
@app.post("/telegram/webhook")
async def telegram_webhook(request: Request):
    data = await request.json()
    update = Update.de_json(data, bot)

    user_id = update.message.from_user.id
    text = update.message.text

    # CASO 1: Si hay respuesta pendiente
    if user_id in pending_responses:
        # Es una respuesta a una pregunta del agente
        future = pending_responses.pop(user_id)
        future.set_result(text)
        return {"ok": True}

    # CASO 2: Nuevo mensaje
    asyncio.create_task(run_agent(user_id, text))
    return {"ok": True}  # Respondemos rápido a Telegram
```

### 4. El agente ejecuta en background

```python
async def run_agent(user_id: int, message: str):
    # Esto puede tomar minutos
    result = await agent.run(
        message,
        deps={'user_id': user_id, 'bot': bot}
    )

    # Enviar respuesta final
    await bot.send_message(chat_id=user_id, text=result.data)
```

## Webhooks vs Polling: Comparación

| Característica | Webhook | Polling |
|---------------|---------|---------|
| **Eficiencia** | Alta | Baja |
| **Latencia** | Inmediata | Delay de polling interval |
| **Uso de recursos** | Bajo | Alto |
| **Complejidad** | Necesita servidor público | Más simple |
| **Ideal para** | Producción | Testing local |

## Problemas Comunes con Webhooks

### Problema 1: Webhook no llega

**Causas**:
- URL incorrecta
- Servidor no es HTTPS (Telegram requiere HTTPS)
- Firewall bloqueando
- Servidor caído

**Solución**:
```bash
# Verificar webhook configurado
curl "https://api.telegram.org/bot<TOKEN>/getWebhookInfo"

# Ver si hay errores
# Si last_error_date > 0, hubo un error
```

### Problema 2: Duplicados

Telegram puede enviar el mismo update múltiples veces si:
- Tu servidor no responde rápido
- Tu servidor devuelve error

**Solución**: Trackear updates procesados:

```python
processed_updates = set()

@app.post("/telegram/webhook")
async def telegram_webhook(request: Request):
    data = await request.json()
    update_id = data.get('update_id')

    if update_id in processed_updates:
        return {"ok": True}  # Ya procesado

    processed_updates.add(update_id)
    # ... procesar ...
```

### Problema 3: Timeout

Tu webhook tarda más de 60 segundos en responder.

**Solución**: Procesar en background:

```python
@app.post("/telegram/webhook")
async def telegram_webhook(request: Request):
    # ... extraer datos ...

    # Procesar en background
    asyncio.create_task(run_agent(user_id, text))

    # Responder inmediatamente
    return {"ok": True}
```

## Testing Webhooks Localmente

Tu localhost no es accesible desde internet. Opciones:

### Opción 1: ngrok (Recomendado)

```bash
# Exponer puerto 8000
ngrok http 8000
```

Te da una URL pública:
```
https://abc123.ngrok.io → http://localhost:8000
```

Configura webhook:
```bash
curl -X POST "https://api.telegram.org/bot<TOKEN>/setWebhook?url=https://abc123.ngrok.io/telegram/webhook"
```

### Opción 2: Polling (para testing)

Usar `getUpdates` en lugar de webhooks:

```python
# No usar en producción
offset = 0
while True:
    updates = telegram.get_updates(offset=offset)
    for update in updates:
        procesar_update(update)
        offset = update.update_id + 1
    time.sleep(1)
```

## Seguridad de Webhooks

### Validar que viene de Telegram

Telegram puede enviar un secreto:

```bash
curl -X POST "https://api.telegram.org/bot<TOKEN>/setWebhook?url=<URL>&secret_token=mi-secreto-super-seguro"
```

En tu servidor:
```python
@app.post("/telegram/webhook")
async def telegram_webhook(request: Request):
    # Verificar token
    token = request.headers.get('X-Telegram-Bot-Api-Secret-Token')
    if token != 'mi-secreto-super-seguro':
        raise HTTPException(status_code=403, detail="Invalid token")

    # ... procesar ...
```

### HTTPS Obligatorio

Telegram **requiere** HTTPS para webhooks en producción.

Localmente:
- ngrok da HTTPS automáticamente

En producción:
- Railway/Fly.io dan HTTPS automáticamente

## Resumen

- **Webhook**: URL que otro servicio llama cuando ocurre un evento
- **Ventajas**: Eficiente, inmediato, bajo uso de recursos
- **Configuración**: Una sola vez con `setWebhook`
- **Funcionamiento**: Telegram → POST → Tu servidor
- **Importante**: Responder rápido (< 60s), procesar en background
- **Testing local**: Usar ngrok
- **Producción**: Railway/Fly.io con HTTPS

Los webhooks son la forma profesional de integrar servicios en tiempo real.
