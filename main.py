from pydantic_ai import Agent
from pydantic_ai.a2a import FastA2A

agent = Agent(name='Potato Agent')
app = FastA2A(agent)

if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host='0.0.0.0', port=8000)
