# ¿Qué son los Futures y Async/Await?

## Analogía Simple: La Pizzería

### Sin Async (Síncrono)

Imagina una pizzería donde el empleado hace UNA cosa a la vez:

```
Cliente 1: "Quiero una pizza"
[Empleado hace la pizza - 10 minutos]
[Empleado entrega pizza]

Cliente 2: "Quiero una pizza"
[Empleado hace la pizza - 10 minutos]
[Empleado entrega pizza]

Cliente 3: "Quiero una pizza"
[Empleado hace la pizza - 10 minutos]
[Empleado entrega pizza]

Total: 30 minutos para 3 pizzas
```

### Con Async (Asíncrono)

Ahora el empleado puede hacer varias cosas mientras las pizzas se cocinan:

```
Cliente 1: "Quiero una pizza"
[Empleado mete pizza 1 al horno]
[MIENTRAS SE COCINA...]

Cliente 2: "Quiero una pizza"
[Empleado mete pizza 2 al horno]
[MIENTRAS SE COCINAN...]

Cliente 3: "Quiero una pizza"
[Empleado mete pizza 3 al horno]

[Después de 10 minutos, todas salen]
[Empleado entrega las 3 pizzas]

Total: 10 minutos para 3 pizzas
```

**Async permite hacer múltiples cosas "al mismo tiempo"** sin bloquear.

## ¿Qué es Async/Await?

### Código Síncrono (Bloqueante)

```python
import time

def hacer_pizza(numero):
    print(f"Haciendo pizza {numero}...")
    time.sleep(10)  # Bloquea por 10 segundos
    print(f"Pizza {numero} lista!")
    return f"Pizza {numero}"

# Hacer 3 pizzas
hacer_pizza(1)  # Espera 10s
hacer_pizza(2)  # Espera 10s
hacer_pizza(3)  # Espera 10s

# Total: 30 segundos
```

**Problema**: El programa se **bloquea** durante `time.sleep()`. No puede hacer nada más.

### Código Asíncrono (No bloqueante)

```python
import asyncio

async def hacer_pizza(numero):
    print(f"Haciendo pizza {numero}...")
    await asyncio.sleep(10)  # NO bloquea
    print(f"Pizza {numero} lista!")
    return f"Pizza {numero}"

async def main():
    # Hacer 3 pizzas en paralelo
    await asyncio.gather(
        hacer_pizza(1),
        hacer_pizza(2),
        hacer_pizza(3)
    )

asyncio.run(main())

# Total: 10 segundos
```

**Ventaja**: `await asyncio.sleep(10)` **no bloquea**. Mientras espera, puede hacer otras cosas.

## Palabras Clave

### `async def`

Define una función asíncrona:

```python
async def mi_funcion():
    return "Hola"
```

**Características**:
- Puede usar `await`
- Devuelve una "coroutine" (promesa)
- Debe ser llamada con `await`

### `await`

Espera a que algo asíncrono termine:

```python
async def obtener_usuario():
    # Esperar a que la base de datos responda
    usuario = await database.get_user(123)
    return usuario
```

**Sin `await`** (error):
```python
async def obtener_usuario():
    # Esto NO funciona
    usuario = database.get_user(123)  # Devuelve coroutine, no el usuario
    return usuario  # Error: no puedes usar coroutines sin await
```

### `asyncio.create_task()`

Ejecuta algo en **background**:

```python
async def tarea_larga():
    await asyncio.sleep(10)
    print("Tarea completada")

async def main():
    # Iniciar tarea en background
    task = asyncio.create_task(tarea_larga())

    # Continuar haciendo otras cosas
    print("Haciendo otras cosas...")
    await asyncio.sleep(1)
    print("Todavía haciendo cosas...")

    # Esperar a que termine (opcional)
    await task

asyncio.run(main())
```

Output:
```
Haciendo otras cosas...
Todavía haciendo cosas...
[Después de 10s]
Tarea completada
```

## ¿Qué es un Future?

Un **Future** es una **promesa** de un valor que llegará en el futuro.

### Analogía: El Ticket del Restaurante

Cuando pides comida, te dan un **ticket** con un número:

```
Ticket #42
"Tu comida estará lista en 15 minutos"
```

El ticket es una **promesa** (Future) de que recibirás comida.

Mientras esperas:
- Puedes sentarte
- Ir al baño
- Ver el celular
- Cuando tu número aparezca en la pantalla, recoges la comida

### Código con Future

```python
import asyncio

async def main():
    # Crear Future (promesa vacía)
    future = asyncio.Future()

    print("Future creado, pero sin valor todavía")
    print(f"¿Tiene valor? {future.done()}")  # False

    # En algún momento, alguien "cumple" la promesa
    future.set_result("¡Aquí está el valor!")

    print(f"¿Tiene valor ahora? {future.done()}")  # True

    # Obtener el valor
    valor = await future
    print(f"Valor: {valor}")  # "¡Aquí está el valor!"

asyncio.run(main())
```

## Future en ask_user

Nuestra tool `ask_user` usa Futures para esperar la respuesta del usuario:

### Paso 1: Crear Future

```python
async def ask_user(ctx: RunContext[dict], question: str) -> str:
    # ...

    # Crear Future (ticket vacío)
    future = asyncio.Future()

    # Guardar Future con clave = user_id
    pending_responses[user_id] = future

    # ...
```

**Estado**: Future creado pero sin valor. Es una promesa de que habrá un valor.

### Paso 2: Esperar con await

```python
    # ...

    # ESPERAR a que alguien cumpla la promesa
    response = await future  # Bloquea aquí

    # ...
```

**¿Qué pasa aquí?**
- El código se **bloquea** en `await future`
- Pero **no bloquea el servidor** (es async)
- Otros requests pueden procesarse
- Cuando el Future reciba un valor, este código continuará

### Paso 3: Cumplir la Promesa

En otro lugar (el webhook):

```python
@app.post("/telegram/webhook")
async def telegram_webhook(request: Request):
    # ...

    if user_id in pending_responses:
        # Obtener Future
        future = pending_responses.pop(user_id)

        # CUMPLIR la promesa
        future.set_result(text)  # "3 días"

    # ...
```

**¿Qué pasa aquí?**
- `future.set_result("3 días")` pone un valor en el Future
- Inmediatamente, el código que estaba esperando (`await future`) se desbloquea
- `response` ahora vale `"3 días"`

### Visualización

```
[Tool ask_user ejecutando]
    ↓
future = asyncio.Future()  # Crear ticket vacío
    ↓
pending_responses[user_id] = future  # Guardar ticket
    ↓
response = await future  # ESPERAR (bloqueado aquí)
    |
    |  [MIENTRAS TANTO...]
    |
    |  [Usuario responde]
    |       ↓
    |  [Webhook recibe respuesta]
    |       ↓
    |  future.set_result("3 días")  # Poner valor en ticket
    |       ↓
    ←------- [DESBLOQUEADO]
    ↓
response = "3 días"  # Tiene el valor
    ↓
return response  # Devolver al agente
```

## Ejemplo Completo: Simulación

Vamos a simular nuestro flujo:

```python
import asyncio

# Diccionario para guardar futures pendientes
pending_responses = {}

async def ask_user(user_id: int, question: str) -> str:
    """Simula la tool ask_user."""
    print(f"[TOOL] Preguntando a usuario {user_id}: {question}")

    # Crear Future
    future = asyncio.Future()
    pending_responses[user_id] = future

    print(f"[TOOL] Esperando respuesta de usuario {user_id}...")

    # ESPERAR (bloquea hasta que haya respuesta)
    response = await future

    print(f"[TOOL] Usuario {user_id} respondió: {response}")

    return response

async def simular_usuario_responde(user_id: int, respuesta: str, delay: int):
    """Simula que el usuario responde después de X segundos."""
    await asyncio.sleep(delay)  # Esperar

    print(f"\n[USUARIO] Usuario {user_id} responde: {respuesta}")

    # Obtener Future y cumplir promesa
    future = pending_responses.pop(user_id)
    future.set_result(respuesta)

async def main():
    user_id = 123

    # Iniciar pregunta y simulación de respuesta en paralelo
    pregunta = ask_user(user_id, "¿Cuántos días?")
    respuesta_usuario = simular_usuario_responde(user_id, "3 días", 3)

    # Esperar a que ambos terminen
    resultado = await asyncio.gather(pregunta, respuesta_usuario)

    print(f"\n[MAIN] Resultado: {resultado[0]}")

asyncio.run(main())
```

Output:
```
[TOOL] Preguntando a usuario 123: ¿Cuántos días?
[TOOL] Esperando respuesta de usuario 123...
[Espera 3 segundos...]
[USUARIO] Usuario 123 responde: 3 días
[TOOL] Usuario 123 respondió: 3 días
[MAIN] Resultado: 3 días
```

## ¿Por qué no Variables Globales?

### Opción A: Variable Global (MAL)

```python
respuesta_usuario = None

async def ask_user(question: str) -> str:
    global respuesta_usuario
    respuesta_usuario = None  # Reset

    # Esperar a que alguien ponga un valor
    while respuesta_usuario is None:
        await asyncio.sleep(0.1)  # Polling (ineficiente)

    return respuesta_usuario
```

**Problemas**:
- Polling (ineficiente)
- No funciona con múltiples usuarios (se sobrescribe)
- Feo

### Opción B: Future (BIEN)

```python
pending_responses = {}  # Diccionario: user_id → future

async def ask_user(user_id: int, question: str) -> str:
    future = asyncio.Future()
    pending_responses[user_id] = future

    # Esperar eficientemente
    response = await future

    return response
```

**Ventajas**:
- No polling (eficiente)
- Funciona con múltiples usuarios (cada uno tiene su future)
- Limpio

## Event Loop

El **event loop** es el "cerebro" que maneja todo el código asíncrono.

### Visualización

```
[Event Loop]
    |
    ├─ [Task 1: ask_user para usuario A]
    |    └─ await future  (esperando)
    |
    ├─ [Task 2: ask_user para usuario B]
    |    └─ await future  (esperando)
    |
    ├─ [Task 3: webhook recibe respuesta de A]
    |    └─ future.set_result("respuesta A")
    |         └─ [Task 1 se desbloquea]
    |
    └─ [Task 4: webhook recibe respuesta de B]
         └─ future.set_result("respuesta B")
              └─ [Task 2 se desbloquea]
```

El event loop **intercambia** entre tasks, ejecutando lo que puede y esperando lo que no puede.

## Async/Await en Nuestro Proyecto

### server.py

```python
@app.post("/telegram/webhook")
async def telegram_webhook(request: Request):  # async
    data = await request.json()  # await

    # ...

    if user_id in pending_responses:
        future = pending_responses.pop(user_id)
        future.set_result(text)  # Cumplir promesa

    # ...

    asyncio.create_task(run_agent(user_id, text))  # Background

    return {"ok": True}

async def run_agent(user_id: int, message: str):  # async
    result = await agent.run(message, deps={...})  # await

    await bot.send_message(chat_id=user_id, text=result.data)  # await
```

### tools.py

```python
async def ask_user(ctx: RunContext[dict], question: str) -> str:  # async
    # ...

    await bot.send_message(...)  # await

    future = asyncio.Future()
    pending_responses[user_id] = future

    response = await future  # await (esperar promesa)

    return response
```

## Cuándo Usar Async

### Usar Async cuando:
- Esperas I/O (red, base de datos, archivos)
- Llamas a APIs
- Esperas respuesta del usuario
- Tienes múltiples tasks concurrentes

### NO usar Async cuando:
- Solo haces cálculos (CPU-bound)
- No esperas nada
- Código simple secuencial

### Ejemplo: NO necesita async

```python
def calcular_suma(a, b):
    return a + b  # Instantáneo, no espera nada
```

### Ejemplo: SÍ necesita async

```python
async def obtener_usuario(user_id):
    user = await database.get_user(user_id)  # Espera I/O
    return user
```

## Resumen

- **Async/Await**: Permite hacer múltiples cosas sin bloquear
- **Future**: Promesa de un valor que llegará en el futuro
- **await future**: Espera a que el Future tenga un valor
- **future.set_result(valor)**: Cumple la promesa, pone el valor
- **asyncio.create_task()**: Ejecuta algo en background
- **Event Loop**: Cerebro que maneja todas las tasks asíncronas

En nuestro proyecto:
1. `ask_user` crea un Future y espera
2. Usuario responde
3. Webhook cumple el Future
4. `ask_user` recibe el valor y continúa

Los Futures son la magia que permite que el agente "espere" respuestas del usuario sin bloquear el servidor.
