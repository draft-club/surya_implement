from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor
import torch

# Charger le modèle et le processeur
model_id = "microsoft/Phi-3.5-vision-instruct"
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype="auto", trust_remote_code=True)
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

# Charger l'image
image_path = "/home/ozdomar/Adil/doc_gpt/vision/data/0001.jpg"  # Remplacez par le chemin réel de votre image
image = Image.open(image_path)

# Créer l'invite pour le modèle
prompt = "<|user|>\n<|image_1|>\nPouvez-vous lire et transcrire le texte de cette décision d'expropriation ?<|end|>\n<|assistant|>\n"

# Préparer les entrées pour le modèle
inputs = processor(images=image, text=prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")

# Générer la réponse
outputs = model.generate(**inputs, max_new_tokens=1000)

# Obtenir l'identifiant du token de padding
pad_token_id = processor.tokenizer.pad_token_id

# Remplacer les valeurs -100 par pad_token_id
outputs = outputs.cpu().numpy()
outputs[outputs == -100] = pad_token_id

# Décoder les sorties modifiées
response = processor.decode(outputs[0], skip_special_tokens=True)

# Afficher la réponse
print(response)

