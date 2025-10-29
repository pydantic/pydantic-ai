# ¿Qué son las Tools?

## Analogía Simple

Imagina que contratas a un asistente personal:

**Sin tools** (solo conocimiento):
```
Tú: ¿Cuánto dinero tengo en el banco?
Asistente: No lo sé, no puedo acceder a tu cuenta.
```

**Con tools** (puede hacer cosas):
```
Tú: ¿Cuánto dinero tengo en el banco?
Asistente: [Usa tool "consultar_banco"]
Asistente: Tienes $1,234.56 en tu cuenta.
```

Las **tools** le dan al agente la capacidad de **hacer cosas** más allá de solo generar texto.

## ¿Qué es una Tool?

Una **tool** (herramienta) es una **función Python** que el agente puede llamar para:
- Obtener información externa (base de datos, APIs)
- Realizar acciones (enviar email, crear archivo)
- Interactuar con el usuario (hacer preguntas)

**Analogía**: Si el agente es un cerebro, las tools son sus manos y sentidos.

## Comparación: Con y Sin Tools

### Sin Tools (Solo texto)

```python
agent = Agent('claude-sonnet-4-20250514')

result = await agent.run("¿Qué tiempo hace en Madrid?")
print(result.data)
# Output: "No tengo acceso a información en tiempo real sobre el clima."
```

El agente solo puede generar texto basado en su conocimiento (que está desactualizado).

### Con Tools (Puede hacer cosas)

```python
async def obtener_clima(ctx: RunContext[dict], ciudad: str) -> str:
    """Obtiene el clima actual de una ciudad."""
    response = requests.get(f"https://api.clima.com/v1/{ciudad}")
    return response.json()['temperatura']

agent = Agent(
    'claude-sonnet-4-20250514',
    tools=[obtener_clima]
)

result = await agent.run("¿Qué tiempo hace en Madrid?")
print(result.data)
# Output: "Actualmente en Madrid hace 22°C con cielo despejado."
```

El agente:
1. Ve que necesita info del clima
2. Llama a la tool `obtener_clima("Madrid")`
3. Recibe el resultado
4. Genera respuesta usando esa info

## Anatomía de una Tool

### Estructura Básica

```python
from pydantic_ai import RunContext

async def nombre_de_tool(ctx: RunContext[dict], parametro: str) -> str:
    """
    Descripción de qué hace la tool.

    Args:
        ctx: Contexto con dependencias
        parametro: Descripción del parámetro

    Returns:
        Descripción de qué devuelve
    """
    # Tu lógica aquí
    resultado = hacer_algo(parametro)
    return resultado
```

### Elementos Importantes

#### 1. Decorador (Opcional)

Puedes registrar tools con decorador:

```python
agent = Agent('claude-sonnet-4-20250514')

@agent.tool
async def buscar_usuario(ctx: RunContext[dict], nombre: str) -> str:
    """Busca un usuario por nombre."""
    return database.find(nombre)
```

O con lista:

```python
async def tool1(ctx, param): ...
async def tool2(ctx, param): ...

agent = Agent(
    'claude-sonnet-4-20250514',
    tools=[tool1, tool2]
)
```

#### 2. Docstring

**MUY IMPORTANTE**: El docstring es lo que el agente lee para decidir si usar la tool.

```python
async def buscar_receta(ctx: RunContext[dict], ingrediente: str) -> str:
    """
    Busca recetas que contengan el ingrediente especificado.

    Args:
        ingrediente: El ingrediente a buscar (ej: "pollo", "tomate")

    Returns:
        Lista de recetas encontradas
    """
```

El modelo de IA lee esto y piensa:
> "Ah, si necesito buscar recetas con un ingrediente, puedo usar esta tool."

**Mal docstring** (vago):
```python
async def buscar(ctx, cosa):
    """Busca algo."""
```

El agente no sabrá cuándo usarla.

**Buen docstring** (específico):
```python
async def buscar_receta(ctx: RunContext[dict], ingrediente: str) -> str:
    """
    Busca recetas de cocina que contengan un ingrediente específico.

    Útil cuando el usuario pregunta cosas como:
    - "¿Qué puedo cocinar con pollo?"
    - "Dame recetas con tomate"

    Args:
        ingrediente: El ingrediente a buscar (ej: "pollo", "tomate", "pasta")

    Returns:
        Lista de hasta 5 recetas con ese ingrediente, incluyendo nombre e instrucciones
    """
```

#### 3. Type Hints

Pydantic AI usa type hints para validar parámetros:

```python
async def calcular(ctx: RunContext[dict], a: int, b: int) -> int:
    """Suma dos números."""
    return a + b
```

Si el modelo intenta llamar con strings, Pydantic AI automáticamente:
1. Intenta convertir: `calcular("5", "3")` → `calcular(5, 3)`
2. Si no puede, lanza error

#### 4. RunContext

El contexto da acceso a dependencias:

```python
async def enviar_email(ctx: RunContext[dict], destinatario: str) -> str:
    # Obtener dependencias
    user_id = ctx.deps['user_id']
    bot = ctx.deps['bot']
    config = ctx.deps['config']

    # Usar dependencias
    await bot.send_message(user_id, f"Email enviado a {destinatario}")
    return "Email enviado"
```

**¿Por qué usar ctx.deps en lugar de variables globales?**
- Más limpio
- Testeable
- No hay conflictos entre múltiples usuarios

## Nuestra Tool: ask_user

### Código Completo

```python
from pydantic_ai import RunContext
import asyncio
from telegram import Bot

# Diccionario global para guardar respuestas pendientes
pending_responses = {}

async def ask_user(ctx: RunContext[dict], question: str) -> str:
    """
    Pregunta algo al usuario vía Telegram y ESPERA la respuesta.

    Args:
        ctx: Contexto con user_id y bot
        question: La pregunta a hacer

    Returns:
        La respuesta del usuario como string
    """
    user_id = ctx.deps['user_id']
    bot = ctx.deps['bot']

    print(f"[TOOL ask_user] Preguntando a usuario {user_id}: {question}")

    # 1. Enviar pregunta por Telegram
    await bot.send_message(chat_id=user_id, text=question)

    # 2. Crear Future para esperar respuesta
    future = asyncio.Future()
    pending_responses[user_id] = future

    print(f"[TOOL ask_user] Esperando respuesta de usuario {user_id}...")

    # 3. ESPERAR la respuesta (bloquea hasta que el usuario responda)
    response = await future

    print(f"[TOOL ask_user] Usuario {user_id} respondió: {response}")

    # 4. Devolver respuesta al agente
    return response
```

### ¿Por qué es especial?

Esta tool es única porque:
1. **Interactúa con el usuario** (la mayoría solo leen/escriben datos)
2. **Espera una respuesta** (puede tomar minutos)
3. **Usa Futures** (ver explicación en 07-que-son-los-futures.md)

## Tipos de Tools

### 1. Tools de Lectura (Read)

Obtienen información:

```python
async def obtener_saldo(ctx: RunContext[dict]) -> str:
    """Obtiene el saldo de la cuenta del usuario."""
    user_id = ctx.deps['user_id']
    saldo = database.get_balance(user_id)
    return f"Tu saldo es: ${saldo}"
```

### 2. Tools de Escritura (Write)

Modifican datos:

```python
async def crear_tarea(ctx: RunContext[dict], titulo: str, fecha: str) -> str:
    """Crea una nueva tarea en el sistema."""
    user_id = ctx.deps['user_id']
    tarea_id = database.create_task(user_id, titulo, fecha)
    return f"Tarea creada con ID {tarea_id}"
```

### 3. Tools de Acción (Action)

Realizan acciones:

```python
async def enviar_notificacion(ctx: RunContext[dict], mensaje: str) -> str:
    """Envía una notificación push al usuario."""
    user_id = ctx.deps['user_id']
    notifier.send(user_id, mensaje)
    return "Notificación enviada"
```

### 4. Tools de Interacción (Interactive)

Interactúan con el usuario (nuestra `ask_user`):

```python
async def ask_user(ctx: RunContext[dict], question: str) -> str:
    """Pregunta algo al usuario y espera respuesta."""
    # ... (código anterior)
```

### 5. Tools de API Externa

Llaman a APIs externas:

```python
async def buscar_en_google(ctx: RunContext[dict], query: str) -> str:
    """Busca información en Google."""
    response = requests.get(
        "https://www.googleapis.com/customsearch/v1",
        params={'q': query, 'key': API_KEY}
    )
    return response.json()['items'][0]['snippet']
```

## Cómo el Agente Decide Usar una Tool

El modelo de IA decide basándose en:

### 1. El System Prompt

```python
SYSTEM_PROMPT = """
Eres un asistente de tareas.

Si el usuario pregunta por sus tareas, USA la tool "obtener_tareas".
Si el usuario quiere crear una tarea, USA la tool "crear_tarea".
"""
```

### 2. El Docstring de la Tool

```python
async def obtener_tareas(ctx: RunContext[dict]) -> str:
    """
    Obtiene todas las tareas pendientes del usuario.

    Usa esta tool cuando el usuario pregunta:
    - "¿Qué tareas tengo?"
    - "Muéstrame mis pendientes"
    - "¿Qué debo hacer hoy?"
    """
```

### 3. El Nombre de la Tool

Nombres descriptivos ayudan:
- ✅ `obtener_clima`, `buscar_usuario`, `crear_tarea`
- ❌ `tool1`, `hacer_cosa`, `x`

### 4. El Contexto de la Conversación

```
Usuario: "Quiero saber el clima"
Agente piensa: "Necesito info del clima → usar tool obtener_clima"
```

## Tools con Múltiples Parámetros

```python
async def buscar_vuelos(
    ctx: RunContext[dict],
    origen: str,
    destino: str,
    fecha: str,
    max_precio: int = 1000
) -> str:
    """
    Busca vuelos disponibles.

    Args:
        origen: Ciudad de origen (ej: "Madrid")
        destino: Ciudad de destino (ej: "Barcelona")
        fecha: Fecha en formato YYYY-MM-DD
        max_precio: Precio máximo en dólares (default: 1000)

    Returns:
        Lista de vuelos disponibles
    """
    vuelos = api.buscar_vuelos(origen, destino, fecha, max_precio)
    return f"Encontré {len(vuelos)} vuelos desde ${vuelos[0]['precio']}"
```

El agente puede llamar:
```python
buscar_vuelos(
    origen="Madrid",
    destino="Barcelona",
    fecha="2025-05-01",
    max_precio=500
)
```

## Tools con Validación

Usar Pydantic para validar entrada:

```python
from pydantic import BaseModel, Field

class BuscarVuelosInput(BaseModel):
    origen: str = Field(min_length=2, max_length=50)
    destino: str = Field(min_length=2, max_length=50)
    fecha: str = Field(pattern=r'\d{4}-\d{2}-\d{2}')  # YYYY-MM-DD
    max_precio: int = Field(gt=0, le=10000)

async def buscar_vuelos(ctx: RunContext[dict], input: BuscarVuelosInput) -> str:
    """Busca vuelos disponibles."""
    # input.origen está validado automáticamente
    vuelos = api.buscar_vuelos(
        input.origen,
        input.destino,
        input.fecha,
        input.max_precio
    )
```

## Error Handling en Tools

### Opción 1: Try/Except

```python
async def obtener_clima(ctx: RunContext[dict], ciudad: str) -> str:
    """Obtiene el clima de una ciudad."""
    try:
        response = requests.get(f"https://api.clima.com/{ciudad}")
        response.raise_for_status()
        return f"El clima en {ciudad} es {response.json()['temp']}°C"
    except requests.exceptions.RequestException:
        return f"No pude obtener el clima de {ciudad}. ¿Está bien escrito el nombre?"
```

### Opción 2: Retries

```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
async def obtener_clima(ctx: RunContext[dict], ciudad: str) -> str:
    """Obtiene el clima de una ciudad."""
    response = requests.get(f"https://api.clima.com/{ciudad}")
    response.raise_for_status()
    return f"El clima en {ciudad} es {response.json()['temp']}°C"
```

Si falla, reintenta automáticamente 3 veces con espera exponencial.

## Tools Síncronas vs Asíncronas

### Síncrona (Blocking)

```python
def buscar_en_db(ctx: RunContext[dict], query: str) -> str:
    """Busca en la base de datos."""
    result = database.query(query)  # Bloquea
    return result
```

**Problema**: Bloquea el servidor mientras espera

### Asíncrona (Non-blocking)

```python
async def buscar_en_db(ctx: RunContext[dict], query: str) -> str:
    """Busca en la base de datos."""
    result = await database.async_query(query)  # No bloquea
    return result
```

**Ventaja**: Permite que otras peticiones se procesen mientras espera

**Recomendación**: Usa async siempre que sea posible.

## Debugging Tools

### Ver cuándo se llaman

```python
async def mi_tool(ctx: RunContext[dict], param: str) -> str:
    """Mi tool."""
    print(f"[DEBUG] mi_tool llamada con param={param}")
    result = hacer_algo(param)
    print(f"[DEBUG] mi_tool devuelve: {result}")
    return result
```

### Ver todas las tools disponibles

```python
agent = Agent('claude-sonnet-4-20250514', tools=TOOLS)

for tool in TOOLS:
    print(f"Tool: {tool.__name__}")
    print(f"Docstring: {tool.__doc__}")
    print()
```

## Resumen

- **Tool**: Función Python que el agente puede llamar
- **Propósito**: Darle capacidades más allá de generar texto
- **Elementos clave**:
  - Docstring (describe qué hace)
  - Type hints (valida parámetros)
  - RunContext (accede a dependencias)
- **Tipos**: Lectura, escritura, acción, interacción, API externa
- **Decisión**: El agente decide cuándo usarlas basándose en docstring y contexto
- **Best practice**: Async, buen docstring, nombres descriptivos

Las tools transforman un chatbot simple en un agente capaz de realizar tareas complejas.
