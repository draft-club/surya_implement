import os
import json
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
from preprocessing import Preprocessor

def perform_multi_page_ocr(image_paths):
    """
    Effectuer l'OCR sur plusieurs pages d'un document contenant un tableau
    et combiner les résultats en un seul JSON.
    """
    # Charger le modèle et le processeur
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

    # Construire le contenu du message avec la structure correcte
    message_content = []
    
    # Ajouter d'abord toutes les images
    for path in image_paths:
        message_content.append({"type": "image", "image": f"file://{path}"})
    
    # Ajouter le texte du prompt
    message_content.append({
        "type": "text",
        "text": "These images are consecutive pages from the same document containing a table that may span multiple pages. "
                "Please analyze all pages and extract the complete table data into a single JSON format. "
                "For each row, include all column values and ensure the structure is consistent across pages. "
                "Combine multiline text into a single value for each cell. "
                "Preserve all text exactly as it appears in the document, including non-Latin scripts, numbers, and symbols. "
                "Do not translate or approximate any text. "
                "Make sure to maintain the correct order of rows across pages."
    })

    # Définir le message pour le modèle avec la structure correcte
    messages = [
        {
            "role": "user",
            "content": message_content
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
    generated_ids = model.generate(**inputs, max_new_tokens=2000)
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
        base_name = os.path.splitext(os.path.basename(image_paths[0]))[0]
        output_file = f"{base_name}_combined.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(json_data, f, ensure_ascii=False, indent=4)
        print(f"Résultat combiné sauvegardé dans : {output_file}")

    except json.JSONDecodeError:
        print("Erreur : la sortie générée n'est pas un JSON valide.")
        print("Sortie brute :", output_text)

if __name__ == "__main__":
    import argparse

    # Modifier le parser pour accepter plusieurs images
    parser = argparse.ArgumentParser(description="Preprocess multiple images and perform OCR.")
    parser.add_argument(
        "image_paths",
        type=str,
        nargs='+',  # Accepte un ou plusieurs arguments
        help="Paths to the image files to process and perform OCR."
    )
    args = parser.parse_args()

    # Prétraiter toutes les images
    preprocessed_paths = []
    for image_path in args.image_paths:
        preprocessed_path = Preprocessor.preprocess_image(image_path)
        print(f"Image prétraitée sauvegardée ici : {preprocessed_path}")
        preprocessed_paths.append(preprocessed_path)

    # OCR sur toutes les images prétraitées
    perform_multi_page_ocr(preprocessed_paths)
