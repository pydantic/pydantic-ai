# Índice de Explicaciones

Esta carpeta contiene guías detalladas explicando cada concepto del proyecto TITAN para personas que no tienen experiencia previa.

## 📚 Guías Disponibles

### 1. [¿Qué es un Servidor?](01-que-es-un-servidor.md)
**Para quién**: Nunca has trabajado con servidores

**Aprenderás**:
- Qué es un servidor y cómo funciona
- Cliente vs Servidor
- Endpoints y URLs
- Puertos y protocolos HTTP
- Analogías simples para entender conceptos

**Tiempo de lectura**: 10 minutos

---

### 2. [¿Qué es FastAPI?](02-que-es-fastapi.md)
**Para quién**: Sabes Python básico, nunca has usado frameworks web

**Aprenderás**:
- Qué es un framework web
- Por qué FastAPI facilita todo
- Decoradores (@app.get, @app.post)
- Request y Response
- Documentación automática
- Comparación con Flask y Node.js

**Tiempo de lectura**: 15 minutos

---

### 3. [¿Qué es Pydantic AI?](03-que-es-pydantic-ai.md)
**Para quién**: Nunca has trabajado con agentes de IA

**Aprenderás**:
- Qué es un agente de IA
- System prompts
- Tools (herramientas)
- RunContext y dependency injection
- Diferencia con usar APIs directamente
- Comparación con LangChain

**Tiempo de lectura**: 20 minutos

---

### 4. [¿Qué son los Webhooks?](04-que-son-los-webhooks.md)
**Para quién**: No entiendes cómo Telegram se comunica con tu servidor

**Aprenderás**:
- Webhook vs Polling
- Cómo funciona un webhook
- Configurar webhook de Telegram
- Debugging de webhooks
- Testing local con ngrok
- Seguridad de webhooks

**Tiempo de lectura**: 15 minutos

---

### 5. [Cómo Funciona el Flujo Completo](05-como-funciona-el-flujo.md)
**Para quién**: Quieres entender el flujo completo del proyecto

**Aprenderás**:
- Paso a paso: desde que el usuario envía un mensaje hasta que recibe respuesta
- Cómo interactúan todos los componentes
- Diagramas de secuencia
- Qué pasa en cada fase
- Logs y debugging

**Tiempo de lectura**: 25 minutos

**Recomendado**: Leer después de 1-4

---

### 6. [¿Qué son las Tools?](06-que-son-las-tools.md)
**Para quién**: Quieres entender cómo el agente puede "hacer cosas"

**Aprenderás**:
- Qué es una tool (herramienta)
- Anatomía de una tool
- Tipos de tools (lectura, escritura, acción, interacción)
- Cómo el agente decide usar una tool
- Nuestra tool `ask_user` explicada
- Error handling en tools

**Tiempo de lectura**: 20 minutos

---

### 7. [¿Qué son los Futures?](07-que-son-los-futures.md)
**Para quién**: No entiendes async/await o cómo esperamos respuestas del usuario

**Aprenderás**:
- Async/Await explicado con analogías
- Qué es un Future (promesa)
- Cómo `ask_user` espera respuestas
- Event loop
- Cuándo usar async
- Código síncrono vs asíncrono

**Tiempo de lectura**: 25 minutos

**Nota**: Este es el concepto más técnico, pero explicado de forma simple

---

### 8. [Deployment Explicado](08-deployment-explicado.md)
**Para quién**: Quieres subir tu proyecto a producción

**Aprenderás**:
- Qué es deployment
- Opciones de deployment (Railway, Fly.io, etc.)
- Paso a paso: deployment con Railway
- Auto-deploy desde GitHub
- Configurar variables de entorno
- Ver logs y debugging
- Costos

**Tiempo de lectura**: 30 minutos

---

## 🗺️ Rutas de Aprendizaje Recomendadas

### Ruta 1: Completo Principiante
**Objetivo**: Entender todo desde cero

1. [¿Qué es un Servidor?](01-que-es-un-servidor.md) - Fundamentos
2. [¿Qué es FastAPI?](02-que-es-fastapi.md) - Framework web
3. [¿Qué son los Webhooks?](04-que-son-los-webhooks.md) - Comunicación
4. [¿Qué es Pydantic AI?](03-que-es-pydantic-ai.md) - Agentes de IA
5. [¿Qué son las Tools?](06-que-son-las-tools.md) - Herramientas del agente
6. [¿Qué son los Futures?](07-que-son-los-futures.md) - Async/Await
7. [Cómo Funciona el Flujo Completo](05-como-funciona-el-flujo.md) - Todo junto
8. [Deployment Explicado](08-deployment-explicado.md) - A producción

**Tiempo total**: ~2.5 horas

---

### Ruta 2: Ya Sé Python Básico
**Objetivo**: Entender conceptos específicos del proyecto

1. [¿Qué es FastAPI?](02-que-es-fastapi.md) - Framework web
2. [¿Qué son los Webhooks?](04-que-son-los-webhooks.md) - Comunicación
3. [¿Qué es Pydantic AI?](03-que-es-pydantic-ai.md) - Agentes de IA
4. [¿Qué son los Futures?](07-que-son-los-futures.md) - Async/Await
5. [Cómo Funciona el Flujo Completo](05-como-funciona-el-flujo.md) - Todo junto
6. [Deployment Explicado](08-deployment-explicado.md) - A producción

**Tiempo total**: ~1.5 horas

---

### Ruta 3: Solo Quiero Entender el Flujo
**Objetivo**: Entender cómo funciona todo junto

1. [Cómo Funciona el Flujo Completo](05-como-funciona-el-flujo.md) - Flujo principal
2. [¿Qué son los Futures?](07-que-son-los-futures.md) - La magia detrás
3. [¿Qué son las Tools?](06-que-son-las-tools.md) - Herramientas

**Tiempo total**: ~1 hora

**Nota**: Recomendado si ya conoces los conceptos básicos

---

### Ruta 4: Solo Deployment
**Objetivo**: Subir el proyecto a producción

1. [Deployment Explicado](08-deployment-explicado.md) - Todo sobre deployment

**Tiempo total**: 30 minutos

**Prerequisito**: El código ya funciona localmente

---

## 🎯 Por Concepto

¿Tienes dudas sobre algo específico?

### Conceptos de Servidor
- [01. ¿Qué es un Servidor?](01-que-es-un-servidor.md)
- [02. ¿Qué es FastAPI?](02-que-es-fastapi.md)
- [04. ¿Qué son los Webhooks?](04-que-son-los-webhooks.md)

### Conceptos de IA
- [03. ¿Qué es Pydantic AI?](03-que-es-pydantic-ai.md)
- [06. ¿Qué son las Tools?](06-que-son-las-tools.md)

### Conceptos de Programación
- [07. ¿Qué son los Futures?](07-que-son-los-futures.md)

### Integración
- [05. Cómo Funciona el Flujo Completo](05-como-funciona-el-flujo.md)

### Producción
- [08. Deployment Explicado](08-deployment-explicado.md)

---

## 💡 Consejos para Estudiar

### 1. No te saltes fundamentos
Si eres principiante, sigue la **Ruta 1** en orden. Cada guía construye sobre la anterior.

### 2. Practica mientras lees
Abre el código del proyecto y sigue las explicaciones.

### 3. Usa las analogías
Cada guía tiene analogías del mundo real. Úsalas para entender conceptos abstractos.

### 4. Experimenta
Cambia cosas en el código y ve qué pasa. Aprenderás más rompiendo y arreglando.

### 5. No te frustres
Estos conceptos pueden ser difíciles al principio. Es normal. Relee las guías si es necesario.

---

## 🆘 ¿Todavía Confundido?

Si después de leer las guías todavía tienes dudas:

1. **Relee la sección específica**: A veces una segunda lectura aclara todo
2. **Ve a la guía de flujo completo**: Ver todo junto puede ayudar
3. **Experimenta con el código**: Cambia cosas y ve qué pasa
4. **Busca en internet**: Googlea el concepto específico
5. **Pregunta**: Busca comunidades de Python/FastAPI/Pydantic AI

---

## 📖 Recursos Adicionales

### Documentación Oficial
- [FastAPI Docs](https://fastapi.tiangolo.com/)
- [Pydantic AI Docs](https://ai.pydantic.dev/)
- [Python Asyncio Docs](https://docs.python.org/3/library/asyncio.html)
- [Telegram Bot API](https://core.telegram.org/bots/api)

### Tutoriales en Video
- [FastAPI Tutorial (YouTube)](https://www.youtube.com/results?search_query=fastapi+tutorial)
- [Python Async/Await (YouTube)](https://www.youtube.com/results?search_query=python+async+await)

### Comunidades
- [FastAPI Discord](https://discord.gg/VQjSZaeJmf)
- [Pydantic Discord](https://discord.gg/pydantic)
- [r/learnpython (Reddit)](https://reddit.com/r/learnpython)

---

¡Buena suerte en tu aprendizaje! 🚀
