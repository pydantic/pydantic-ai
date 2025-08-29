from pydantic_ai import Agent, NativeOutput
from pydantic_ai.models.outlines import OutlinesModel
from pydantic_ai.settings import ModelSettings
from pydantic import BaseModel


class Box(BaseModel):
    width: int
    height: int
    depth: int
    units: int


def transformers_example():

    print("---- start transformers_example ----")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    hf_model = AutoModelForCausalLM.from_pretrained("erwanf/gpt2-mini")
    hf_tokenizer = AutoTokenizer.from_pretrained("erwanf/gpt2-mini")
    chat_template = '{% for message in messages %}{{ message.role }}: {{ message.content }}{% endfor %}'
    hf_tokenizer.chat_template = chat_template

    model = OutlinesModel.from_transformers(hf_model, hf_tokenizer, settings=ModelSettings(max_new_tokens=100))
    agent = Agent(model, output_type=[Box])

    response = agent.run_sync('Give me the dimensions of a box')
    print("response.output: ", response.output)

    response2 = agent.run_sync('Give me another box', message_history=response.all_messages())
    print("response2.output: ", response2.output)

    print("all_messages: ", response2.all_messages())

    print("---- end transformers_example ----")


def llama_cpp_example():
    print("---- start llama_cpp_example ----")

    from llama_cpp import Llama

    llama_model = Llama.from_pretrained(
        repo_id="TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
        filename="tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        n_ctx=2048,  # 2K context window
    )

    model = OutlinesModel.from_Somllama_cpp(llama_model)
    agent = Agent(model, output_type=NativeOutput([Box]))

    response = agent.run_sync('Give me the dimensions of a box')
    print("response.output: ", response.output)

    response2 = agent.run_sync('Give me another box', message_history=response.all_messages())
    print("response2.output: ", response2.output)

    print("all_messages: ", response2.all_messages())

    print("---- end llama_cpp_example ----")


if __name__ == "__main__":
    transformers_example()
    #llama_cpp_example()
    #existing()
