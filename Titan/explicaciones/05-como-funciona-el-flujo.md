# Cómo Funciona el Flujo Completo

## Visión General

Vamos a seguir un mensaje desde que el usuario lo escribe hasta que recibe la respuesta.

```
[Usuario] → [Telegram] → [Webhook] → [FastAPI] → [Agent] → [Tools]
                                                               ↓
[Usuario] ← [Telegram] ← [Respuesta] ← [Agent] ← [Tool Result]
```

## Ejemplo Paso a Paso

Vamos a seguir esta conversación:

```
Usuario: "Quiero un plan de entrenamiento"
Bot: "¿Cuántos días a la semana puedes entrenar?"
Usuario: "3 días"
Bot: "¿Qué equipo tienes?"
Usuario: "Mancuernas y barra"
Bot: [Genera plan completo]
```

---

## FASE 1: Usuario envía mensaje inicial

### Paso 1.1: Usuario escribe en Telegram

```
Usuario escribe: "Quiero un plan de entrenamiento"
```

### Paso 1.2: Telegram envía webhook a tu servidor

```http
POST https://tu-servidor.com/telegram/webhook
Content-Type: application/json

{
  "update_id": 123456789,
  "message": {
    "message_id": 1,
    "from": {
      "id": 987654,
      "first_name": "Juan"
    },
    "text": "Quiero un plan de entrenamiento"
  }
}
```

### Paso 1.3: FastAPI recibe el webhook

En `server.py`:

```python
@app.post("/telegram/webhook")
async def telegram_webhook(request: Request):
    # Recibir datos
    data = await request.json()
    update = Update.de_json(data, bot)

    # Extraer información
    user_id = update.message.from_user.id  # 987654
    text = update.message.text  # "Quiero un plan de entrenamiento"

    print(f"[WEBHOOK] Usuario {user_id} dice: {text}")
```

**Log en terminal**:
```
==================================================
[WEBHOOK] Mensaje de usuario 987654: Quiero un plan de entrenamiento
==================================================
```

### Paso 1.4: Verificar si es respuesta a pregunta pendiente

```python
# CASO 1: ¿Hay una pregunta pendiente?
if user_id in pending_responses:
    # No hay (es el primer mensaje)
    pass

# CASO 2: ¿Ya hay conversación activa?
if user_id in active_conversations:
    # No hay
    pass

# CASO 3: Nuevo mensaje
active_conversations.add(user_id)  # Marcar como activo
```

### Paso 1.5: Iniciar agente en background

```python
# Ejecutar agente SIN esperar (en background)
asyncio.create_task(run_agent(user_id, text))

# Responder a Telegram inmediatamente
return {"ok": True, "message": "Agent started"}
```

**¿Por qué en background?**
- El agente puede tomar minutos (hace múltiples preguntas)
- Telegram espera respuesta en < 60 segundos
- Si no respondes rápido, Telegram reintenta el webhook

---

## FASE 2: Agente ejecuta

### Paso 2.1: Función run_agent inicia

```python
async def run_agent(user_id: int, message: str):
    print(f"[AGENT] Ejecutando agente para usuario {user_id}...")

    # Ejecutar agente con deps
    result = await agent.run(
        message,  # "Quiero un plan de entrenamiento"
        deps={'user_id': user_id, 'bot': bot}
    )
```

**Log en terminal**:
```
[AGENT] Ejecutando agente para usuario 987654...
```

### Paso 2.2: Pydantic AI envía mensaje a Claude

Detrás de escena, Pydantic AI hace esto:

```http
POST https://api.anthropic.com/v1/messages
{
  "model": "claude-sonnet-4-20250514",
  "max_tokens": 4096,
  "system": "Eres un entrenador personal...",
  "messages": [
    {"role": "user", "content": "Quiero un plan de entrenamiento"}
  ],
  "tools": [
    {
      "name": "ask_user",
      "description": "Pregunta algo al usuario y espera respuesta...",
      "input_schema": {
        "type": "object",
        "properties": {
          "question": {"type": "string"}
        }
      }
    }
  ]
}
```

### Paso 2.3: Claude responde con tool call

Claude piensa:
> "Necesito saber cuántos días puede entrenar. Voy a usar la tool ask_user."

Claude responde:
```json
{
  "role": "assistant",
  "content": [
    {
      "type": "tool_use",
      "id": "toolu_123",
      "name": "ask_user",
      "input": {
        "question": "¿Cuántos días a la semana puedes entrenar?"
      }
    }
  ]
}
```

### Paso 2.4: Pydantic AI ejecuta la tool

Pydantic AI ve el tool call y ejecuta:

```python
# Llamar a ask_user
result = await ask_user(
    ctx=RunContext(deps={'user_id': 987654, 'bot': bot}),
    question="¿Cuántos días a la semana puedes entrenar?"
)
```

---

## FASE 3: Tool ask_user ejecuta

### Paso 3.1: Enviar pregunta al usuario

En `tools.py`:

```python
async def ask_user(ctx: RunContext[dict], question: str) -> str:
    user_id = ctx.deps['user_id']  # 987654
    bot = ctx.deps['bot']

    print(f"[TOOL ask_user] Preguntando a usuario {user_id}: {question}")

    # Enviar mensaje por Telegram
    await bot.send_message(
        chat_id=user_id,
        text=question  # "¿Cuántos días a la semana puedes entrenar?"
    )
```

**Log en terminal**:
```
[TOOL ask_user] Preguntando a usuario 987654: ¿Cuántos días a la semana puedes entrenar?
```

**Usuario ve en Telegram**:
```
Bot: ¿Cuántos días a la semana puedes entrenar?
```

### Paso 3.2: Crear Future y esperar

```python
    # Crear Future (promesa de respuesta futura)
    future = asyncio.Future()

    # Guardar Future con clave = user_id
    pending_responses[user_id] = future

    print(f"[TOOL ask_user] Esperando respuesta de usuario {user_id}...")

    # ESPERAR (bloquea hasta que el usuario responda)
    response = await future

    print(f"[TOOL ask_user] Usuario {user_id} respondió: {response}")

    return response
```

**Estado actual**:
- El agente está **esperando**
- `pending_responses[987654]` tiene un Future pendiente
- La ejecución está bloqueada en `await future`

**Log en terminal**:
```
[TOOL ask_user] Esperando respuesta de usuario 987654...
```

---

## FASE 4: Usuario responde

### Paso 4.1: Usuario escribe respuesta

```
Usuario escribe: "3 días"
```

### Paso 4.2: Telegram envía webhook

```http
POST https://tu-servidor.com/telegram/webhook
{
  "update_id": 123456790,
  "message": {
    "message_id": 2,
    "from": {"id": 987654},
    "text": "3 días"
  }
}
```

### Paso 4.3: FastAPI recibe webhook

```python
@app.post("/telegram/webhook")
async def telegram_webhook(request: Request):
    data = await request.json()
    update = Update.de_json(data, bot)

    user_id = update.message.from_user.id  # 987654
    text = update.message.text  # "3 días"

    print(f"[WEBHOOK] Mensaje de usuario {user_id}: {text}")
```

**Log en terminal**:
```
==================================================
[WEBHOOK] Mensaje de usuario 987654: 3 días
==================================================
```

### Paso 4.4: Detectar respuesta pendiente

```python
    # CASO 1: ¿Hay una pregunta pendiente?
    if user_id in pending_responses:
        print(f"[WEBHOOK] Usuario {user_id} tiene respuesta pendiente. Resolviendo...")

        # Obtener el Future
        future = pending_responses.pop(user_id)

        # CUMPLIR la promesa con la respuesta
        future.set_result(text)  # "3 días"

        return {"ok": True, "message": "Response forwarded to agent"}
```

**Log en terminal**:
```
[WEBHOOK] Usuario 987654 tiene respuesta pendiente. Resolviendo...
```

**¿Qué acaba de pasar?**
- El webhook resolvió el Future con `"3 días"`
- La tool `ask_user` (que estaba esperando) ahora recibe el valor
- ¡El agente continúa!

---

## FASE 5: Agente continúa con la respuesta

### Paso 5.1: ask_user recibe la respuesta

En `tools.py`, la línea que estaba bloqueada:

```python
    # Esto estaba esperando...
    response = await future

    # ¡Ahora recibe "3 días"!
    print(f"[TOOL ask_user] Usuario {user_id} respondió: {response}")

    return response  # Devolver "3 días" al agente
```

**Log en terminal**:
```
[TOOL ask_user] Usuario 987654 respondió: 3 días
```

### Paso 5.2: Pydantic AI envía tool result a Claude

```http
POST https://api.anthropic.com/v1/messages
{
  "messages": [
    {"role": "user", "content": "Quiero un plan de entrenamiento"},
    {
      "role": "assistant",
      "content": [
        {
          "type": "tool_use",
          "id": "toolu_123",
          "name": "ask_user",
          "input": {"question": "¿Cuántos días..."}
        }
      ]
    },
    {
      "role": "user",
      "content": [
        {
          "type": "tool_result",
          "tool_use_id": "toolu_123",
          "content": "3 días"
        }
      ]
    }
  ]
}
```

### Paso 5.3: Claude decide hacer otra pregunta

Claude piensa:
> "OK, 3 días. Ahora necesito saber qué equipo tiene."

Claude responde con otro tool call:
```json
{
  "role": "assistant",
  "content": [
    {
      "type": "tool_use",
      "id": "toolu_456",
      "name": "ask_user",
      "input": {
        "question": "¿Qué equipo tienes disponible?"
      }
    }
  ]
}
```

### Paso 5.4: Se repite el ciclo

**FASE 3 de nuevo**:
- Tool envía: "¿Qué equipo tienes disponible?"
- Crea Future
- Espera...

**Usuario responde**: "Mancuernas y barra"

**FASE 4 de nuevo**:
- Webhook recibe "Mancuernas y barra"
- Resuelve Future
- Tool devuelve respuesta

---

## FASE 6: Respuesta final

### Paso 6.1: Claude tiene toda la información

Claude ahora sabe:
- Días: 3
- Equipo: Mancuernas y barra

Claude piensa:
> "Ya tengo todo. Voy a generar el plan."

Claude responde **sin tool call** (respuesta final):
```json
{
  "role": "assistant",
  "content": [
    {
      "type": "text",
      "text": "**PLAN DE ENTRENAMIENTO**\n\n📊 **Perfil:**\n- Días: 3\n- Equipo: Mancuernas y barra\n\n🏋️ **Rutina:**\n..."
    }
  ]
}
```

### Paso 6.2: Pydantic AI devuelve resultado

```python
    # En run_agent()
    result = await agent.run(...)

    print(f"[AGENT] Agente terminó para usuario {user_id}")
    print(f"[AGENT] Respuesta: {result.data}")
```

**Log en terminal**:
```
[AGENT] Agente terminó para usuario 987654
[AGENT] Respuesta: **PLAN DE ENTRENAMIENTO**...
```

### Paso 6.3: Enviar respuesta al usuario

```python
    # Enviar respuesta final
    await bot.send_message(
        chat_id=user_id,
        text=result.data
    )

    print(f"[AGENT] Respuesta enviada a usuario {user_id}")
```

**Usuario ve en Telegram**:
```
Bot: **PLAN DE ENTRENAMIENTO**

📊 **Perfil:**
- Días: 3
- Equipo: Mancuernas y barra
- Nivel: Intermedio

🏋️ **Rutina:**
[Plan completo...]
```

### Paso 6.4: Limpiar

```python
    finally:
        # Remover de conversaciones activas
        active_conversations.discard(user_id)
        print(f"[AGENT] Conversación cerrada para usuario {user_id}\n")
```

**Log en terminal**:
```
[AGENT] Respuesta enviada a usuario 987654
[AGENT] Conversación cerrada para usuario 987654
```

---

## Diagrama de Secuencia Completo

```
Usuario          Telegram       Webhook        Agent         Tool
  |                |              |              |             |
  |--"Plan"------->|              |              |             |
  |                |--POST------->|              |             |
  |                |              |--run()------>|             |
  |                |              |              |--ask_user()->|
  |                |              |              |             |--send("¿Días?")-->Telegram
  |                |              |              |             |
  |                |              |              |             |--await future
  |<--"¿Días?"-----|<-------------|<-------------|<------------|
  |                |              |              |             |
  |--"3 días"----->|              |              |             |
  |                |--POST------->|              |             |
  |                |              |--resolve---->|------------>|
  |                |              |              |<--"3 días"--|
  |                |              |              |             |
  |                |              |              |--ask_user()->|
  |                |              |              |             |--send("¿Equipo?")-->Telegram
  |<--"¿Equipo?"---|<-------------|<-------------|<------------|
  |                |              |              |             |
  |--"Mancuernas"->|              |              |             |
  |                |--POST------->|              |             |
  |                |              |--resolve---->|------------>|
  |                |              |              |<--"Manc."---|
  |                |              |              |             |
  |                |              |              |--generate-->|
  |                |              |<--result-----|             |
  |                |              |--send------->Telegram      |
  |<--"PLAN"-------|<-------------|              |             |
```

---

## Resumen del Flujo

1. **Usuario envía mensaje** → Telegram → Webhook
2. **Webhook** → Inicia agente en background
3. **Agente** → Piensa y decide usar tool `ask_user`
4. **Tool** → Envía pregunta a Telegram y crea Future
5. **Tool espera** (await future)
6. **Usuario responde** → Telegram → Webhook
7. **Webhook** → Resuelve Future con respuesta
8. **Tool** → Recibe respuesta y la devuelve al agente
9. **Agente** → Repite 3-8 hasta tener toda la info
10. **Agente** → Genera respuesta final
11. **Servidor** → Envía respuesta a usuario vía Telegram

La magia está en los **Futures**: permiten que el agente "espere" una respuesta del usuario sin bloquear todo el servidor.
