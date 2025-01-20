# pip install transformers datasets diffusers

from transformers import pipeline
from diffusers import DiffusionPipeline
from datasets import load_dataset

#pipeline allows for easy access to models from the Hugging Face model hub
#diffusers allows for easy access to models from the Diffusion model hub
#datasets allows for easy access to datasets from the Hugging Face

#Testattavia malleja:
# salimmk/Pori
# Finnish-NLP/Ahma-3B
# https://huggingface.co/LumiOpen

#my_pipeline = pipeline("text-generation", model="Finnish-NLP/Ahma-3B")
#result = my_pipeline("Moro, mitä kuuluu, ymmärrätkö Suomea?", max_length=50)
#print(result)

from transformers import pipeline

generator = pipeline("text-generation", model="LumiOpen/viking-7b", tokenizer="LumiOpen/viking-7b")
print("Aloitetaan generointi...")
response = generator("Translate this text to Finnish: Hello, how are you?", max_length=100)
print("Generointi valmis.")
print(response)