import argparse
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import json
import os
import cv2
import numpy as np
from PIL import Image, ImageEnhance




# Fonctions de preprocessing
def enhance_contrast_and_binarize(image_path):
    """
    Améliorer le contraste et binariser l'image.
    """
    img = Image.open(image_path)
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(2.0)
    img = img.convert("L")
    img = img.point(lambda x: 0 if x < 128 else 255, '1')
    temp_path = "temp_contrast.jpg"
    img.save(temp_path)
    return temp_path

def remove_noise(image_path):
    """
    Supprimer le bruit de l'image.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    denoised = cv2.bilateralFilter(img, 9, 75, 75)
    temp_path = "temp_denoised.jpg"
    cv2.imwrite(temp_path, denoised)
    return temp_path

def deskew_with_hough_transform(image_path, output_path):
    """
    Corriger l'inclinaison d'une image en utilisant la transformée de Hough.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    edges = cv2.Canny(binary, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
    angles = []
    if lines is not None:
        for rho, theta in lines[:, 0]:
            angle = (theta * 180 / np.pi) - 90
            angles.append(angle)
    if not angles:
        print("Aucune ligne détectée. L'image pourrait ne pas être inclinée.")
        cv2.imwrite(output_path, img)
        return output_path
    median_angle = np.median(angles)
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    cv2.imwrite(output_path, rotated)
    return output_path

def preprocess_image(image_path):
    """
    Pipeline complet de prétraitement.
    """
    contrast_path = enhance_contrast_and_binarize(image_path)
    denoised_path = contrast_path
    directory, filename = os.path.split(image_path)
    preprocessed_filename = f"preprocessed_{filename}"
    preprocessed_path = os.path.join(directory, preprocessed_filename)
    final_path = deskew_with_hough_transform(denoised_path, preprocessed_path)
    if os.path.exists(contrast_path):
        os.remove(contrast_path)
    if os.path.exists(denoised_path):
        os.remove(denoised_path)
    return final_path

# Fonction principale pour OCR
def perform_ocr(image_path):
    """
    Effectuer l'OCR sur une image prétraitée.
    """
    # Charger le modèle et le processeur
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

    # Définir les messages avec l'image et la demande en JSON
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"file://{image_path}"},
                {
                    "type": "text",
                    "text": "Analyze the provided image."
                            "If it contains a table, extract its content and transcribe it into JSON format with the following specifications:"
                            "Each column in the table should be used as a key in the JSON object."
                            "The corresponding row values should be mapped to these keys."
                            "If a cell contains multiple lines of text, combine them into a single value."
                            "Include and preserve all full names exactly as they appear in the table."
                            "Retain the text in its original format, including non-Latin scripts, numbers, and symbols, without translation or approximation."
                            "Generate a valid JSON structure that faithfully represents the table’s content."
                            "If no table is detected in the image, return an empty JSON object like this: {}."

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
    generated_ids = model.generate(**inputs, max_new_tokens=1000)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    # Nettoyer le formatage pour obtenir un JSON pur
    try:
        cleaned_json = output_text.strip("```json").strip("```").strip()
        json_data = json.loads(cleaned_json)

        # Sauvegarder dans un fichier JSON
        output_file = "output.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(json_data, f, ensure_ascii=False, indent=4)
        print(f"JSON sauvegardé dans : {output_file}")
    except json.JSONDecodeError:
        print("Erreur : la sortie générée n'est pas un JSON valide.")
        print("Sortie brute :", output_text)

if __name__ == "__main__":
    # Ajouter un argument pour le chemin de l'image
    parser = argparse.ArgumentParser(description="Preprocess an image and perform OCR.")
    parser.add_argument(
        "image_path",
        type=str,
        help="Path to the image file to process and perform OCR."
    )
    args = parser.parse_args()

    # Prétraitement
    preprocessed_path = preprocess_image(args.image_path)
    print(f"Image prétraitée sauvegardée ici : {preprocessed_path}")

    # OCR sur l'image prétraitée
    perform_ocr(preprocessed_path)

