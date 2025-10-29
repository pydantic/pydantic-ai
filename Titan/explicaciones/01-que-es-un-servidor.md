# ¿Qué es un Servidor?

## Analogía Simple

Imagina que tienes un **restaurante**:
- El restaurante está **abierto 24/7**
- Los clientes (usuarios) llegan y hacen pedidos (peticiones)
- Los cocineros (tu código) preparan la comida (procesan la petición)
- Los meseros entregan la comida (respuestas)

**Un servidor es exactamente eso**, pero en el mundo digital.

## ¿Qué hace nuestro servidor?

Nuestro servidor Python:

1. **Está siempre escuchando** en una dirección (URL):
   ```
   https://tu-servidor.com/telegram/webhook
   ```

2. **Recibe mensajes** de Telegram cuando un usuario escribe

3. **Ejecuta código** Python (tu agente de IA)

4. **Devuelve respuestas** al usuario vía Telegram

## Cliente vs Servidor

### Cliente (Tu teléfono)
- **Hace peticiones**: "Oye servidor, el usuario dijo 'Hola'"
- Espera respuestas
- Muestra resultados al usuario

### Servidor (Tu código en la nube)
- **Recibe peticiones**: "OK, recibí 'Hola'"
- Procesa la información (ejecuta el agente)
- **Envía respuestas**: "Respóndele: ¿En qué puedo ayudarte?"

## ¿Dónde vive el servidor?

Cuando haces "deployment", tu servidor vive en:
- **Localmente** (tu computadora): Solo para testing
- **Railway/Fly.io** (la nube): Producción, 24/7

## Componentes de Nuestro Servidor

```
[Internet] ← → [Servidor]
                  ↓
            [FastAPI]  ← Framework web
                  ↓
            [Tu código Python]
                  ↓
            [Agent de IA]
```

## URLs y Endpoints

Un servidor tiene diferentes **endpoints** (puntos de entrada):

```python
@app.get("/")              # https://tu-servidor.com/
@app.get("/health")        # https://tu-servidor.com/health
@app.post("/telegram/webhook")  # https://tu-servidor.com/telegram/webhook
```

Piensa en ellos como **diferentes puertas de tu restaurante**:
- Puerta principal: `/`
- Puerta de empleados: `/health`
- Puerta de entregas: `/telegram/webhook`

## ¿Por qué necesitamos un servidor?

**Problema**: Quieres que tu bot de Telegram funcione 24/7, pero no puedes dejar tu laptop encendida siempre.

**Solución**: Pones tu código en un servidor en la nube que está siempre encendido.

## Tipos de Servidores

### 1. Servidor Web (HTTP)
- Nuestro caso
- Recibe peticiones HTTP
- Devuelve respuestas HTTP

### 2. Servidor de Base de Datos
- Guarda y recupera datos
- Ejemplo: PostgreSQL, MongoDB

### 3. Servidor de Archivos
- Almacena archivos
- Ejemplo: AWS S3

## Puerto

Un servidor escucha en un **puerto** específico:

```python
port = 8000  # Nuestro servidor escucha en el puerto 8000
```

Analogía: Si tu servidor es un edificio, el puerto es el número de apartamento.

```
https://tu-servidor.com:8000/telegram/webhook
                        ↑
                     Puerto
```

En producción (Railway), el puerto suele ser 80 (HTTP) o 443 (HTTPS), por lo que no necesitas especificarlo.

## Protocolo HTTP

Nuestro servidor usa **HTTP** (HyperText Transfer Protocol):

### Métodos HTTP:
- **GET**: "Dame información" (leer)
  ```python
  @app.get("/health")  # Solo lectura
  ```

- **POST**: "Aquí te envío información" (crear/procesar)
  ```python
  @app.post("/telegram/webhook")  # Recibe datos
  ```

- **PUT**: "Actualiza esto" (actualizar)
- **DELETE**: "Borra esto" (eliminar)

## ¿Qué pasa cuando alguien accede a tu servidor?

### Ejemplo: Usuario envía mensaje en Telegram

1. **Usuario**: Escribe "Hola" en Telegram
2. **Telegram**: Envía POST request a tu servidor:
   ```
   POST https://tu-servidor.com/telegram/webhook
   Body: {"message": {"text": "Hola", "from": {"id": 123456}}}
   ```
3. **Tu servidor** (server.py):
   - Recibe la petición
   - Extrae el mensaje y user_id
   - Ejecuta el agente
4. **Agente**: Procesa "Hola" y decide responder
5. **Tu servidor**: Envía respuesta de vuelta a Telegram
6. **Telegram**: Muestra la respuesta al usuario

## Analogía Completa

```
Usuario → [Telegram App] → [Internet] → [Tu Servidor] → [Agente de IA]
                                                              ↓
                                                         Respuesta
                                                              ↓
Usuario ← [Telegram App] ← [Internet] ← [Tu Servidor] ← [Agente de IA]
```

Es como enviar una carta:
1. Escribes la carta (mensaje)
2. La pones en el buzón (Telegram)
3. El cartero la lleva (Internet)
4. Llega a la dirección (tu servidor)
5. Alguien la lee y responde (agente)
6. La respuesta vuelve por el mismo camino

## Resumen

- **Servidor**: Programa que está siempre corriendo y responde a peticiones
- **Cliente**: Quien hace peticiones (Telegram, navegador, app)
- **Endpoint**: URL específica que hace algo (`/telegram/webhook`)
- **Puerto**: Número donde el servidor escucha (8000)
- **HTTP**: Protocolo de comunicación (GET, POST, etc.)

En nuestro proyecto, el servidor es el **intermediario** entre Telegram y tu agente de IA.
