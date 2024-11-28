import os
import json
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
from preprocessing import Preprocessor  # Importer la classe de preprocessing

def perform_ocr_with_prompt(image_path):
    """
    Effectuer l'OCR avec un prompt explicite pour extraire les tableaux en JSON
    et afficher le résultat à l'écran ainsi que le sauvegarder dans un fichier JSON.
    """
    # Charger le modèle et le processeur
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="cuda",
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

    # Définir le message pour le modèle
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"file://{image_path}"},
                {
                    "type": "text",
                    "text": "Please analyze the provided image. If it contains a table, transcribe it into JSON format. "
                            "For each row, include all column values and ensure the structure is clear. Combine multiline text into a single value for each cell. "
                            "Preserve all text exactly as it appears in the image, including non-Latin scripts, numbers, and symbols. "
                            "Do not translate or approximate any text."
                },
            ],
        }
    ]

    # Préparation des entrées
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

    # Inférence avec le modèle
    generated_ids = model.generate(**inputs, max_new_tokens=1000)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    # Nettoyer et afficher le résultat
    try:
        cleaned_json = output_text.strip("```json").strip("```").strip()
        json_data = json.loads(cleaned_json)

        # Afficher le JSON au format lisible
        print(json.dumps(json_data, ensure_ascii=False, indent=4))

        # Sauvegarder le résultat dans un fichier JSON
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_file = f"{base_name}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(json_data, f, ensure_ascii=False, indent=4)
        print(f"Résultat sauvegardé dans : {output_file}")

    except json.JSONDecodeError:
        print("Erreur : la sortie générée n'est pas un JSON valide.")
        print("Sortie brute :", output_text)

if __name__ == "__main__":
    import argparse

    # Ajouter un argument pour le chemin de l'image
    parser = argparse.ArgumentParser(description="Preprocess an image and perform OCR.")
    parser.add_argument(
        "image_path",
        type=str,
        help="Path to the image file to process and perform OCR."
    )
    args = parser.parse_args()

    # Prétraitement
    preprocessed_path = Preprocessor.preprocess_image(args.image_path)
    print(f"Image prétraitée sauvegardée ici : {preprocessed_path}")

    # OCR sur l'image prétraitée
    perform_ocr_with_prompt(preprocessed_path)

