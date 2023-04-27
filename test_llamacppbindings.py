from llama_cpp import Llama
llm = Llama(model_path="./llama.cpp/models/ggml-model-f16.bin")
output = llm("### Instruction: Write a Python program to perform insertion sort on a list. ### Response:", max_tokens=256, stop=["Output"], echo=True)


print(output)
