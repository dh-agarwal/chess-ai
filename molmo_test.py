from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("allenai/MolmoE-1B-0924", trust_remote_code=True)
processor = AutoProcessor.from_pretrained(
    'allenai/MolmoE-1B-0924',
    trust_remote_code=True,
    torch_dtype='auto',
    device_map='auto'
)
# Load image and ask about its content
from PIL import Image
image = Image.open("image.png").convert("RGB")
question = "What is this image of?"

# Prompt the model with the image and question
from PIL import Image
import requests
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig

url = "https://github.com/haotian-liu/LLaVA/blob/1a91fc274d7c35a9b50b3cb29c4247ae5837ce39/images/llava_v1_5_radar.jpg?raw=true"
image = Image.open(requests.get(url, stream=True).raw)
prompt = "[INST] <image>\nWhat is shown in this image? [/INST]"

inputs = processor(prompt, image, return_tensors="pt")

output = model.generate_from_batch(
    inputs,
    GenerationConfig(max_new_tokens=200, stop_strings="<|endoftext|>"),
    tokenizer=processor.tokenizer
)

# only get generated tokens; decode them to text
generated_tokens = output[0,inputs['input_ids'].size(1):]
generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

# print the generated text
print(generated_text)
output = model.generate(**inputs, max_new_tokens=100)
