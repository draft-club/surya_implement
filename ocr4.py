from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import json

# Charger le modèle
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)

# Charger le processeur
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

# Définir les messages avec l'image et la demande en JSON
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "file:///home/ozdomar/Adil/doc_gpt/vision/data/test-new4.jpg"},
            {
                "type": "text",
                "text": "Please read the table from the provided image and transcribe it into JSON format. "
                        "Each row of the table should be represented as a JSON object with clear keys for each column. "
                        "Ensure that all details are accurately captured and the JSON is well-formatted."
            },
        ],
    }
]

# Préparation pour l'inférence
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to("cuda")

# Inférence : génération de l'output
generated_ids = model.generate(**inputs, max_new_tokens=500)
generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)[0]

# Nettoyer le formatage pour obtenir un JSON pur
try:
    # Supprimer les balises éventuelles de markdown (```json) et charger le JSON
    cleaned_json = output_text.strip("```json").strip("```").strip()
    json_data = json.loads(cleaned_json)

    # Afficher un JSON bien formaté
    print(json.dumps(json_data, ensure_ascii=False, indent=4))

except json.JSONDecodeError as e:
    print("Erreur lors de la conversion en JSON :", e)

