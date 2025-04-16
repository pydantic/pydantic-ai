from pydantic_ai import Agent
from pydantic_ai.a2a import FastA2A
from pydantic_ai.a2a.task_manager import InMemoryTaskManager

agent = Agent(name='Potato Agent')
task_manager = InMemoryTaskManager(agent)
app = FastA2A(agent, task_manager=task_manager)

if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host='0.0.0.0', port=8000)
