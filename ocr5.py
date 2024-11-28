import argparse
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import json

def clean_and_combine_lines(json_data):
    """
    Combine lines for cells with multiple rows into a single string.
    """
    for entry in json_data:
        if "الاسم الكامل" in entry:
            # Vérifier et combiner les noms séparés
            names = entry["الاسم الكامل"].split(" - ")
            entry["الاسم الكامل"] = " - ".join([name.strip() for name in names if name.strip()])
    return json_data

def main(image_path):
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
                {"type": "image", "image": f"file://{image_path}"},
                {
                    "type": "text",
                    "text": "If the provided image contains a table, transcribe it into JSON format. "
                            "Each row of the table should be represented as a JSON object with clear keys for each column. "
                            "Ensure that all details are accurately captured, including multiple lines within a single cell, and the JSON is well-formatted."
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

        # Vérifier et combiner les lignes manquantes
        json_data = clean_and_combine_lines(json_data)

        # Afficher un JSON bien formaté
        print(json.dumps(json_data, ensure_ascii=False, indent=4))

        # Sauvegarder dans un fichier JSON si nécessaire
        output_file = "output.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(json_data, f, ensure_ascii=False, indent=4)
        print(f"JSON sauvegardé dans : {output_file}")
    except json.JSONDecodeError:
        # Si le texte généré n'est pas un JSON valide
        print("Aucun tableau détecté ou réponse non JSON. Aucun output généré.")

# Point d'entrée du script
if __name__ == "__main__":
    # Définir l'argument pour le chemin de l'image
    parser = argparse.ArgumentParser(description="Process an image to extract table content in JSON format.")
    parser.add_argument(
        "image_path",
        type=str,
        help="Path to the image file to be processed.",
    )
    args = parser.parse_args()

    # Exécuter la fonction principale avec le chemin fourni
    main(args.image_path)

