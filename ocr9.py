import argparse
from importlib.resources import contents

from qwen_vl_utils import process_vision_info

from transformers import Qwen2VLForConditionalGeneration, AutoProcessor



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
    img = enhancer.enhance(10.0)
    img = img.convert("L")
    # img = img.point(lambda x: 0 if x < 100 else 255, '1')
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
    #(h, w) = img.shape[:2]
    #center = (w // 2, h // 2)
    #M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    #rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    cv2.imwrite(output_path, img)
    return output_path



def preprocess_image(root_path):
    """
    Complete preprocessing pipeline for all images in the root_path.

    This function processes each image in the root_path by:
    - Enhancing contrast and binarizing the image.
    - Denoising the image.
    - Deskewing the image with a Hough transform.

    Args:
    - root_path (str): The directory containing images to be processed.

    Returns:
    - List of paths to the processed images.
    """
    # Directory to store processed images
    processed_dir = "processed_dir"
    os.makedirs(processed_dir, exist_ok=True)  # Ensure directory exists

    images_path_list = []

    # Iterate through all supported image files in the root_path
    supported_extensions = (".png", ".jpg", ".jpeg")
    for filename in os.listdir(root_path):
        if filename.lower().endswith(supported_extensions):
            print(f"Processing: {filename}")
            image_path = os.path.join(root_path, filename)


            try:
                # Step 1: Enhance contrast and binarize
                contrast_path = enhance_contrast_and_binarize(image_path)

                # Step 2: Denoising
                denoised_path = contrast_path

                # Step 3: Deskew with Hough transform
                preprocessed_filename = f"preprocessed_{filename}"
                preprocessed_path = os.path.join(processed_dir, preprocessed_filename)
                #final_path= preprocessed_path
                final_path = deskew_with_hough_transform(denoised_path, preprocessed_path)

                # Add the processed image path to the list
                images_path_list.append(final_path)
                print(f"Saved processed image: {final_path}")

                # Clean up intermediate files
                # if os.path.exists(contrast_path):
                #     os.remove(contrast_path)
                # if os.path.exists(denoised_path):
                #     os.remove(denoised_path)

            except Exception as e:
                print(f"Error processing {filename}: {e}")

    print(f"Total processed images: {len(images_path_list)}")
    return images_path_list


# Fonction principale pour OCR
def perform_ocr(processed_path_list):
    """
    Effectuer l'OCR sur une liste d'images prétraitées.
    """
    # Charger le modèle et le processeur
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")



    # Définir les messages avec l'image et la demande en JSON

    image_dicts_list = [{"type": "image", "image": f"file://{image_path}"} for image_path in processed_path_list]
    print(image_dicts_list)

    prompt_text_dict =   {
                    "type": "text",
                    "text": "If the provided image contains a table, transcribe it into JSON format. "
                            "Reply in arabic , Each row of the table should be represented as a JSON object with clear keys for each column. "
                            "Ensure that all details are accurately captured, including multiple lines within a single cell, and the JSON is well-formatted."
                }

    # prompt_text_dict = {"type":"text",
    #                     "text": "Please analyze the image provided and extract the table data if a table is present. "
    #                             "Answer in arabic and Format the extracted table data as a JSON object where:"
    #                             "Each key in the JSON is a row number, starting from 1 for the first row."
    #                             "Each row should be represented by an object, where the keys are the column names of the table and the values are the corresponding data in that row."
    #                             "If any value in a row has multiple lines, separate each line by a comma."
    #                             "Please only return the JSON object without any extra text or explanations."}

    # Join all elements in the list into a single string, with each dictionary represented as a string, separated by commas
    image_dicts = ', '.join([str(image_dict) for image_dict in image_dicts_list])

    content = image_dicts_list

    content.append(prompt_text_dict)



    messages = [{"role": "user",
                  "content": content , }]



    # Print the result



    # Préparation pour l'inférence
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)

    image_inputs,video_inputs= process_vision_info(messages)
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
    model = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

    print("Model and processor loaded successfully.")

    # Prétraitement
    root_path = './yassine_data'
    root_path = './amine_data'
    preprocessed_path_list = preprocess_image(root_path)



    # OCR sur l'image prétraitée
    perform_ocr(preprocessed_path_list)

