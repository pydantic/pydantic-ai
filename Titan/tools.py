"""
Tools que el agente puede usar.
La tool ask_user es la que hace preguntas al usuario.
"""
from pydantic_ai import RunContext
import asyncio
from telegram import Bot

# Diccionario global para guardar respuestas pendientes
# Clave: user_id, Valor: Future que espera respuesta
pending_responses = {}

async def ask_user(ctx: RunContext[dict], question: str) -> str:
    """
    Tool que pregunta al usuario vía Telegram y ESPERA la respuesta.

    Args:
        ctx: Contexto con información del usuario
        question: La pregunta que queremos hacer

    Returns:
        La respuesta del usuario como string
    """
    user_id = ctx.deps['user_id']
    bot = ctx.deps['bot']

    print(f"[TOOL ask_user] Preguntando a usuario {user_id}: {question}")

    # Enviar pregunta por Telegram
    await bot.send_message(
        chat_id=user_id,
        text=question
    )

    # Crear un Future para esperar la respuesta
    future = asyncio.Future()
    pending_responses[user_id] = future

    print(f"[TOOL ask_user] Esperando respuesta de usuario {user_id}...")

    # ESPERAR la respuesta (esto bloquea hasta que el usuario responda)
    response = await future

    print(f"[TOOL ask_user] Usuario {user_id} respondió: {response}")

    return response

# Lista de tools disponibles
TOOLS = [
    ask_user
]
