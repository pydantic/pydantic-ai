# ¿Qué es Pydantic AI?

## Analogía Simple

Imagina que quieres contratar a un **asistente personal**:

**Sin Pydantic AI**: Tendrías que:
- Aprender la API de OpenAI (compleja)
- Aprender la API de Anthropic (diferente)
- Aprender la API de Google (otra diferente)
- Manejar errores manualmente
- Validar respuestas manualmente
- Programar reintentos
- Todo desde cero...

**Con Pydantic AI**: Te dan:
- Una interfaz unificada para todos los modelos
- Manejo automático de errores
- Validación de respuestas
- Sistema de tools integrado
- Dependency injection
- ¡Todo listo para usar!

## ¿Qué es Pydantic AI?

Pydantic AI es una **biblioteca de Python** que hace fácil trabajar con agentes de IA.

Características principales:
1. **Interfaz unificada**: Mismo código para Claude, GPT, Gemini, etc.
2. **Tools**: El agente puede usar funciones Python
3. **Type-safe**: Usa tipos de Python para validar todo
4. **Dependency Injection**: Pasa contexto al agente fácilmente
5. **Streaming**: Recibe respuestas en tiempo real

## Comparación: Con y Sin Pydantic AI

### SIN Pydantic AI (usando Anthropic directamente)

```python
import anthropic

client = anthropic.Anthropic(api_key="...")

# Hacer una pregunta
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "Hola"}
    ]
)

# Manejar respuesta
text = response.content[0].text

# Para tools, necesitas MUCHO más código...
# Y si quieres cambiar a OpenAI, tienes que reescribir todo
```

### CON Pydantic AI

```python
from pydantic_ai import Agent

agent = Agent('claude-sonnet-4-20250514')

result = await agent.run("Hola")
print(result.data)  # Respuesta

# ¡Y funciona igual con OpenAI!
agent = Agent('openai:gpt-4')
```

Mucho más simple.

## Conceptos Clave

### 1. Agent

El **Agent** es tu asistente de IA:

```python
from pydantic_ai import Agent

agent = Agent(
    'claude-sonnet-4-20250514',  # Modelo
    system_prompt="Eres un chef experto",  # Instrucciones
    tools=[hacer_receta],  # Tools que puede usar
    retries=2  # Reintentar si falla
)
```

**Modelos soportados**:
- Claude: `'claude-sonnet-4-20250514'`
- OpenAI: `'openai:gpt-4'`
- Google: `'google:gemini-1.5-pro'`
- Groq: `'groq:llama-3'`
- Y más...

### 2. System Prompt

Las instrucciones que le das al agente:

```python
SYSTEM_PROMPT = """
Eres un entrenador personal.
Tu trabajo es crear planes de entrenamiento.

REGLAS:
- Siempre pregunta días disponibles
- Siempre pregunta equipo disponible
- Genera un plan detallado
"""

agent = Agent(
    'claude-sonnet-4-20250514',
    system_prompt=SYSTEM_PROMPT
)
```

Piensa en ello como el "manual de empleado" que le das a tu asistente.

### 3. Tools (Herramientas)

**Tools** son funciones Python que el agente puede llamar:

```python
from pydantic_ai import RunContext

async def buscar_receta(ctx: RunContext[dict], ingrediente: str) -> str:
    """
    Busca una receta con el ingrediente especificado.
    """
    # Tu lógica aquí
    recetas = database.buscar(ingrediente)
    return f"Encontré {len(recetas)} recetas con {ingrediente}"

agent = Agent(
    'claude-sonnet-4-20250514',
    tools=[buscar_receta]
)
```

**¿Cómo funciona?**
1. Usuario: "Dame una receta con pollo"
2. Agente piensa: "Necesito buscar recetas con pollo"
3. Agente llama a `buscar_receta("pollo")`
4. Función devuelve resultado
5. Agente usa el resultado para responder

### 4. RunContext (Contexto)

El **RunContext** da información a tus tools:

```python
async def enviar_email(ctx: RunContext[dict], destinatario: str) -> str:
    user_id = ctx.deps['user_id']  # Obtener user_id del contexto
    bot = ctx.deps['bot']  # Obtener bot del contexto

    await bot.send_message(chat_id=user_id, text=f"Email enviado a {destinatario}")
    return "Email enviado"
```

**¿Por qué es útil?**
- Puedes pasar información que las tools necesitan
- No tienes que hacer variables globales
- Es más limpio y testeable

### 5. Ejecutar el Agente

```python
# Ejecutar agente
result = await agent.run(
    "Quiero un plan de entrenamiento",
    deps={'user_id': 123456, 'bot': bot}
)

# Obtener respuesta
print(result.data)
```

**Parámetros**:
- Primer argumento: El mensaje del usuario
- `deps`: Dependencias que pasas al contexto (para tools)

## Nuestro agent.py Explicado

```python
from pydantic_ai import Agent
from tools import TOOLS

# System prompt con instrucciones
SYSTEM_PROMPT = """
Eres un entrenador personal experto en fitness.

Tu trabajo es crear planes de entrenamiento personalizados.

IMPORTANTE:
- Siempre necesitas saber: días disponibles, equipo, nivel, y objetivo del usuario
- Si no tienes esta información, USA la tool "ask_user" para preguntarle
- Puedes hacer múltiples preguntas hasta tener toda la información
- Una vez tengas todo, genera un plan detallado
"""

# Crear agente
def create_agent():
    agent = Agent(
        'claude-sonnet-4-20250514',  # Modelo de Claude
        system_prompt=SYSTEM_PROMPT,  # Instrucciones
        tools=TOOLS,  # [ask_user]
        retries=2  # Si falla, reintenta 2 veces
    )
    return agent

# Instancia global
agent = create_agent()
```

## Tools en Detalle

### Definir una Tool

```python
from pydantic_ai import RunContext

async def ask_user(ctx: RunContext[dict], question: str) -> str:
    """
    Pregunta algo al usuario y espera respuesta.

    Args:
        ctx: Contexto con información del usuario
        question: La pregunta

    Returns:
        La respuesta del usuario
    """
    user_id = ctx.deps['user_id']
    bot = ctx.deps['bot']

    # Enviar pregunta
    await bot.send_message(chat_id=user_id, text=question)

    # Esperar respuesta (ver explicación de Futures)
    # ...

    return respuesta
```

**Elementos importantes**:
1. **Docstring**: Pydantic AI se lo pasa al modelo para que sepa qué hace la tool
2. **Type hints**: `question: str` → La pregunta es un string
3. **Return type**: `-> str` → La tool devuelve un string
4. **ctx: RunContext[dict]**: Contexto con dependencias

### Registrar Tools

```python
TOOLS = [
    ask_user,
    buscar_en_db,
    enviar_notificacion
]

agent = Agent(
    'claude-sonnet-4-20250514',
    tools=TOOLS
)
```

### ¿Cómo el Agente Decide Usar una Tool?

El modelo decide basándose en:
1. El **docstring** de la tool
2. Los **nombres de los parámetros**
3. El **system prompt**
4. El **contexto de la conversación**

Ejemplo:

```python
async def buscar_usuario(ctx: RunContext[dict], nombre: str) -> str:
    """
    Busca un usuario por nombre en la base de datos.

    Args:
        nombre: El nombre del usuario a buscar

    Returns:
        Información del usuario encontrado
    """
    # ...
```

Usuario: "¿Existe un usuario llamado Juan?"

El agente piensa:
- "Necesito buscar en la base de datos"
- "Tengo una tool llamada `buscar_usuario`"
- "El docstring dice que busca usuarios por nombre"
- "Voy a llamarla con nombre='Juan'"

## Dependency Injection

Pasar datos a las tools sin usar variables globales:

```python
# En server.py
result = await agent.run(
    mensaje,
    deps={
        'user_id': 123456,
        'bot': bot,
        'database': db,
        'config': config
    }
)

# En tools.py
async def mi_tool(ctx: RunContext[dict]) -> str:
    user_id = ctx.deps['user_id']
    bot = ctx.deps['bot']
    db = ctx.deps['database']
    # ...
```

**Ventajas**:
- No hay variables globales
- Fácil de testear
- Más limpio

## Streaming (Opcional)

Recibir la respuesta en tiempo real:

```python
async with agent.run_stream("Cuéntame una historia") as result:
    async for chunk in result.stream():
        print(chunk, end='', flush=True)
```

Output:
```
Había una vez...
una princesa...
que vivía en...
un castillo...
```

## Error Handling

Pydantic AI maneja errores automáticamente:

```python
agent = Agent(
    'claude-sonnet-4-20250514',
    retries=2  # Reintenta 2 veces si falla
)
```

Si el modelo falla:
1. Intento 1: Falla
2. Espera 1 segundo
3. Intento 2: Falla
4. Espera 2 segundos
5. Intento 3: Si falla, lanza excepción

Puedes manejar excepciones:

```python
try:
    result = await agent.run("Hola")
except Exception as e:
    print(f"Error: {e}")
```

## Output Types

Puedes forzar que el agente devuelva un formato específico:

```python
from pydantic import BaseModel

class Plan(BaseModel):
    dias: int
    ejercicios: list[str]
    duracion_minutos: int

agent = Agent(
    'claude-sonnet-4-20250514',
    output_type=Plan
)

result = await agent.run("Dame un plan de 3 días")
print(result.data.dias)  # 3
print(result.data.ejercicios)  # ['Sentadillas', 'Flexiones', ...]
```

El modelo **siempre** devolverá un objeto `Plan` válido.

## Mensajes y Conversaciones

Ver el historial de mensajes:

```python
result = await agent.run("Hola")

# Ver todos los mensajes intercambiados
for message in result.all_messages():
    print(f"{message.role}: {message.content}")
```

Output:
```
user: Hola
assistant: ¡Hola! ¿En qué puedo ayudarte?
```

Si el agente usó tools:
```
user: Dame un plan
assistant: [llama a tool ask_user]
tool: ¿Cuántos días?
user: 3 días
assistant: [genera plan]
```

## Diferencia con LangChain

### LangChain (más complejo)

```python
from langchain.chat_models import ChatAnthropic
from langchain.agents import initialize_agent
from langchain.tools import Tool

llm = ChatAnthropic(...)
tools = [Tool(...), Tool(...)]
agent = initialize_agent(tools, llm, agent="structured-chat-zero-shot-react-description")
result = agent.run("Hola")
```

### Pydantic AI (más simple)

```python
from pydantic_ai import Agent

agent = Agent('claude-sonnet-4-20250514', tools=[...])
result = await agent.run("Hola")
```

**Diferencias**:
- Pydantic AI es más simple
- Pydantic AI usa async/await nativo
- Pydantic AI tiene mejor integración con Pydantic (type safety)
- LangChain tiene más features avanzados

## Resumen

- **Pydantic AI**: Biblioteca para crear agentes de IA en Python
- **Agent**: Tu asistente de IA configurado
- **System Prompt**: Instrucciones para el agente
- **Tools**: Funciones que el agente puede llamar
- **RunContext**: Contexto con dependencias para las tools
- **Dependency Injection**: Pasar datos sin variables globales
- **Type Safety**: Usa tipos de Python para validar todo

Pydantic AI hace que trabajar con agentes de IA sea tan fácil como escribir funciones de Python normales.
