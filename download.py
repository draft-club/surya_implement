from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

model_id = "microsoft/Phi-3.5-vision-instruct"
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype="auto", trust_remote_code=True)
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

