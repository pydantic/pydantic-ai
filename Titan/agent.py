"""
Agente de IA con Pydantic AI.
Este agente tiene acceso a la tool ask_user para hacer preguntas.
"""
from pydantic_ai import Agent
from tools import TOOLS
import os

# System prompt del agente
SYSTEM_PROMPT = """
Eres un entrenador personal experto en fitness.

Tu trabajo es crear planes de entrenamiento personalizados.

IMPORTANTE:
- Siempre necesitas saber: días disponibles, equipo, nivel, y objetivo del usuario
- Si no tienes esta información, USA la tool "ask_user" para preguntarle
- Puedes hacer múltiples preguntas hasta tener toda la información
- Una vez tengas todo, genera un plan detallado

FORMATO DE RESPUESTA FINAL:
Cuando tengas toda la info, responde con:

**PLAN DE ENTRENAMIENTO**

📊 **Perfil:**
- Días: X
- Equipo: Y
- Nivel: Z
- Objetivo: W

🏋️ **Rutina:**
[Detalles del plan]

Sé conciso pero completo.
"""

# Crear agente
def create_agent():
    """
    Crea y configura el agente de Pydantic AI.

    Returns:
        Agente configurado con tools y system prompt
    """
    agent = Agent(
        'claude-sonnet-4-20250514',
        system_prompt=SYSTEM_PROMPT,
        tools=TOOLS,
        retries=2
    )

    return agent

# Instancia global del agente
agent = create_agent()
