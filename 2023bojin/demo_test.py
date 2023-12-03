import langchain
from langchain.llms import CTransformers

config = {'max_new_tokens': 256, 'repetition_penalty': 1.1, 'temperature': 0, 'context_length': 10000}
#https://github.com/marella/ctransformers#config For config of CTransformers

llm=OpenAI(temperature=0)

llm = CTransformers(model="TheBloke/CodeLlama-7B-Instruct-GGUF", 
                    model_file="codellama-7b-instruct.Q4_K_M.gguf",config=config, verbose=True)