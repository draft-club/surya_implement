import argparse
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import json
import os
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import pytesseract





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

def extract_text_from_image(image_path):
        try:
            # Open the image file
            img = Image.open(image_path)

            # Perform OCR using pytesseract
            text = pytesseract.image_to_string(img, lang='ara',config='--psm 3 --oem 3 -c classify_bln_numeric_mode=1')

            return text
        except Exception as e:
            return f"Error: {e}"
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





#     prompt_message = """
# Analyze the provided image and perform the following tasks:
#
# 1. If the image contains **text only**, extract and return the Arabic text exactly as it appears in the image, preserving all formatting, punctuation, diacritics (if present), and symbols.
# 2. If the image contains a **table only**, return `{NO text found}`.
# 3. If the image contains **both text and tables**, extract and return only the text content in Arabic, completely omitting any data or information from the table.
#
# ### Guidelines:
# - The text should be preserved in Arabic exactly as it appears, without any modification, translation, or interpretation.
# - If the text includes special formatting, symbols, or text in other languages, they should be retained in the output as is.
# - Completely ignore any information contained within tables and exclude it from the output.
# """
#     prompt_message_3_4 = """
# Analyze the provided image and determine which of the following cases it represents:
#
# Case Definitions:
# 1. Case 1: The image contains paragraphs of text only.
# 2. Case 2: The image contains both paragraphs of text and a table.
# 3. Case 3: The image contains a table only.
#
# Additional Notes:
# - Images may contain headers and footers that are not part of the main content. Ensure these are excluded during extraction.
# - Any paragraphs of text within the boundaries of a table should be considered part of the table and omitted from text extraction.
#
# Output Requirements:
# Based on the identified case, perform the following actions:
# - Case 1: Extract and return the paragraphs of text exactly as they appear in the image, omitting any headers, footers, or text within tables.
# - Case 2: Extract and return only the paragraphs of text outside the table, omitting headers, footers, and any content (including paragraphs) within tables.
# - Case 3: Return the message: "{No text in this image}".
#
# Output Format:
# 1. A clean and contextually relevant output reflecting the extracted text or the specified message based on the identified case.
# 2. Ensure that paragraphs within tables are not treated as standalone text and are omitted from extraction.
#
# Additional Guidance:
# - Clearly differentiate between main content, headers, footers, and tables (including text within tables).
# - Focus on extracting only the central content outside tables that fits the identified case criteria.
# """
#works for 4001_part3
    # prompt_message = """Analyze the given image and extract its main content in arabic into HTML format."""
# This prompt works for 0001, 0002 ( adds numbers in the text 1982, 2022), 0003 (adds header info+numbers of ra9m lmarsoum not complete)
# 0003_cropped (additional llm message), 0004 ( llm identification of the case is wrong but retreives right txt + 64326 number multiple times )
# 0005 (header + blank lines), 0006 ( header + table elements ), 1001 (text from inside the table)
# 1002 ( header +errors in dates mai to mars+doesn't extract numerical amount+ hallucination maroc-maroc-maroc in arabic), 1002_cropped xxx ( nothing extracted )
# 3001 xxx ( nothing extracted, remark : barcode in the middle of the image ), 4001 xxx (information incomplete)
# 5001 xxx (information incomplete + repetition of the same txt),
# 5001 with enhaner 20.0 ( inversed information )
# xxx means doesn't work
    prompt_message = """
Analyze the given image and follow these steps:

1. Determine if the image contains:
   - Only textual paragraphs, or
   - Both textual paragraphs and tables.
   
2. Identify and extract only the textual paragraphs from the image, ignoring any content that is part of a table.

3. Output the extracted textual paragraphs in plain text format, with each paragraph separated by a blank line.

"""
#     prompt_message_soso = """
# Analyze the given image and extract all standalone arabic paragraphs of text (not inside tables). Do not extract any text that is part of a table. Return the extracted text as plain text, with each paragraph separated by a blank line.
#
# Instructions:
# 1. Ignore any text located within tables.
# 2. Extract only the standalone text outside of tables.
# 3. Format the output with one paragraph per line, separated by blank lines.
#
# Return the extracted paragraphs exactly as they appear in the image, formatted as plain text.
# """
#     prompt_message = """
# Analyze the provided image and determine which of the following cases it represents:
#
# Case Definitions:
# 1. Case 1: The image contains paragraphs of text only.
# 2. Case 2: The image contains both paragraphs of text and a table.
# 3. Case 3: The image contains a table only.
# 4. Case 4: The image is empty (contains nothing).
#
# Output Requirements:
# 1. Confidence Score:
#     - Return a table listing the confidence score (a probability value between 0 and 1) for each case.
#     - The table should have two columns: "Case" and "Confidence Score".
#     - Ensure the sum of all confidence scores equals 1.
#
# 2. Based on the identified case, perform the following actions:
#     - Case 1: Extract and return the paragraphs of text exactly as they appear in the image.
#     - Case 2: Extract and return only the paragraphs of text, omitting any content from the table.
#     - Case 3: Return the message: "{No text in this image}".
#     - Case 4: Return "{The image is empty}".
#
# Output Format:
# 1. A table of confidence scores for each case.
# 2. The extracted text or the appropriate message based on the identified case.
#
# Notes:
# - Ensure the confidence scoring mechanism accounts for visible textual and tabular content accurately.
# - If multiple cases seem equally likely, use a tie-breaking heuristic (e.g., prioritizing the presence of text over emptiness).
# - Include any assumptions or additional details that influenced the confidence scoring.
# """

    # Définir les messages avec l'image et la demande en JSON
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"file://{image_path}"},
                {
                    "type": "text",
                    "text":  prompt_message               },
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
    print (output_text)



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
    # perform_ocr(preprocessed_path)
    extracted_text = extract_text_from_image(preprocessed_path)
    print("\nExtracted Text:\n")
    print(extracted_text)
