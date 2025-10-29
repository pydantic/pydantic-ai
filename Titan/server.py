"""
Servidor FastAPI que recibe webhooks de Telegram
y ejecuta el agente con Pydantic AI.
"""
from fastapi import FastAPI, Request
from telegram import Update, Bot
from agent import agent
from tools import pending_responses
import asyncio
import os
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Configuración
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# Verificar que tenemos las variables
if not TELEGRAM_BOT_TOKEN or not ANTHROPIC_API_KEY:
    raise ValueError("Faltan variables de entorno. Revisa tu archivo .env")

# Configurar Anthropic API
os.environ["ANTHROPIC_API_KEY"] = ANTHROPIC_API_KEY

# Crear bot de Telegram
bot = Bot(token=TELEGRAM_BOT_TOKEN)

# Crear app FastAPI
app = FastAPI(title="TITAN Agent Server")

# Diccionario para trackear conversaciones activas
# Evita que se inicien múltiples agentes para el mismo usuario
active_conversations = set()

@app.get("/")
async def root():
    """Endpoint raíz para verificar que el servidor está vivo"""
    return {"status": "ok", "message": "TITAN Agent Server is running"}

@app.get("/health")
async def health():
    """Health check para deployment"""
    return {"status": "healthy"}

@app.post("/telegram/webhook")
async def telegram_webhook(request: Request):
    """
    Webhook que recibe updates de Telegram.

    Flujo:
    1. Recibe mensaje de Telegram
    2. Si hay respuesta pendiente → la resuelve
    3. Si no → inicia nuevo agente
    """
    try:
        # Parsear update de Telegram
        data = await request.json()
        update = Update.de_json(data, bot)

        # Extraer mensaje
        if not update.message or not update.message.text:
            return {"ok": True, "message": "No text message"}

        message = update.message
        user_id = message.from_user.id
        text = message.text

        print(f"\n{'='*50}")
        print(f"[WEBHOOK] Mensaje de usuario {user_id}: {text}")
        print(f"{'='*50}\n")

        # CASO 1: Hay una respuesta pendiente
        if user_id in pending_responses:
            print(f"[WEBHOOK] Usuario {user_id} tiene respuesta pendiente. Resolviendo...")
            future = pending_responses.pop(user_id)
            future.set_result(text)
            return {"ok": True, "message": "Response forwarded to agent"}

        # CASO 2: Conversación ya activa para este usuario
        if user_id in active_conversations:
            print(f"[WEBHOOK] Conversación activa para {user_id}. Ignorando mensaje duplicado.")
            return {"ok": True, "message": "Conversation already active"}

        # CASO 3: Nuevo mensaje → iniciar agente
        print(f"[WEBHOOK] Iniciando agente para usuario {user_id}...")
        active_conversations.add(user_id)

        # Ejecutar agente en background
        asyncio.create_task(run_agent(user_id, text))

        return {"ok": True, "message": "Agent started"}

    except Exception as e:
        print(f"[ERROR] Error en webhook: {e}")
        import traceback
        traceback.print_exc()
        return {"ok": False, "error": str(e)}

async def run_agent(user_id: int, message: str):
    """
    Ejecuta el agente de Pydantic AI.

    Esta función corre en background y puede tomar tiempo
    (porque el agente puede hacer múltiples preguntas).
    """
    try:
        print(f"[AGENT] Ejecutando agente para usuario {user_id}...")

        # Ejecutar agente con deps (user_id y bot)
        result = await agent.run(
            message,
            deps={'user_id': user_id, 'bot': bot}
        )

        print(f"[AGENT] Agente terminó para usuario {user_id}")
        print(f"[AGENT] Respuesta: {result.data}")

        # Enviar respuesta final
        await bot.send_message(
            chat_id=user_id,
            text=result.data
        )

        print(f"[AGENT] Respuesta enviada a usuario {user_id}")

    except Exception as e:
        print(f"[ERROR] Error ejecutando agente para {user_id}: {e}")
        import traceback
        traceback.print_exc()

        # Enviar mensaje de error al usuario
        await bot.send_message(
            chat_id=user_id,
            text="❌ Lo siento, ocurrió un error. Por favor intenta de nuevo."
        )

    finally:
        # Remover de conversaciones activas
        active_conversations.discard(user_id)
        print(f"[AGENT] Conversación cerrada para usuario {user_id}\n")

async def notify_n8n(user_id: int, response: str):
    """
    (OPCIONAL) Notifica a n8n cuando se completa una conversación.

    Descomenta y configura si quieres integrar con n8n.
    """
    import httpx
    # Reemplaza con tu webhook de n8n
    N8N_WEBHOOK_URL = os.getenv("N8N_WEBHOOK_URL")

    if not N8N_WEBHOOK_URL:
        return

    try:
        async with httpx.AsyncClient() as client:
            await client.post(
                N8N_WEBHOOK_URL,
                json={
                    "user_id": user_id,
                    "response": response,
                    "timestamp": asyncio.get_event_loop().time()
                }
            )
            print(f"[N8N] Notificación enviada para usuario {user_id}")
    except Exception as e:
        print(f"[ERROR] Error notificando a n8n: {e}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    print(f"\n🚀 Iniciando servidor en puerto {port}...")
    print(f"📍 Webhook URL: http://localhost:{port}/telegram/webhook\n")
    uvicorn.run(app, host="0.0.0.0", port=port)
