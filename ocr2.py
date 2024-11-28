from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor
import torch

# Identifiant du modèle
model_id = "microsoft/Phi-3.5-vision-instruct"

# Charger le modèle
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype="auto",
    trust_remote_code=True,
    _attn_implementation='flash_attention_2'  # Remplacez par 'eager' si flash_attn n'est pas installé
)

# Charger le processeur
processor = AutoProcessor.from_pretrained(
    model_id,
    trust_remote_code=True,
    num_crops=16  # Utilisez 16 pour une image unique
)

# Chemin vers votre image locale
image_path = "/home/ozdomar/Adil/doc_gpt/vision/data/0001.jpg"  # Remplacez par le chemin réel de votre image
image = Image.open(image_path)

# Préparer le prompt
prompt = "<|user|>\n<|image_1|>\nاقرأ النص الموجود في هذه الصورة وانسخه<|end|>\n<|assistant|>\n" 
#prompt = "<|user|>\n<|image_1|>\nPouvez-vous lire et transcrire le texte de cette image ?<|end|>\n<|assistant|>\n"

# Appliquer le template de chat
messages = [{"role": "user", "content": prompt}]
formatted_prompt = processor.tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

# Préparer les entrées
inputs = processor(
    formatted_prompt,
    images=image,
    return_tensors="pt"
).to("cuda" if torch.cuda.is_available() else "cpu")

# Paramètres de génération
generation_args = {
    "max_new_tokens": 1000,
    "temperature": 5.0,
    "do_sample": False,
}

# Générer la réponse
generate_ids = model.generate(
    **inputs,
    eos_token_id=processor.tokenizer.eos_token_id,
    **generation_args
)

# Supprimer les tokens d'entrée
generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]

# Décoder la réponse
response = processor.batch_decode(
    generate_ids,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False
)[0]

print(response)

