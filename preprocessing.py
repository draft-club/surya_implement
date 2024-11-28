import cv2
import numpy as np
from PIL import Image, ImageEnhance
import os
import argparse

def enhance_contrast_and_binarize(image_path):
    """
    Améliorer le contraste et binariser l'image.
    """
    # Charger l'image avec Pillow
    img = Image.open(image_path)

    # Améliorer le contraste
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(2.0)  # Augmenter le contraste (facteur 2.0)

    # Convertir en niveaux de gris et binariser
    img = img.convert("L")  # Convertir en niveaux de gris
    img = img.point(lambda x: 0 if x < 128 else 255, '1')  # Binarisation

    # Sauvegarder l'image temporairement pour traitement OpenCV
    temp_path = "temp_contrast.jpg"
    img.save(temp_path)
    return temp_path

def remove_noise(image_path):
    """
    Supprimer le bruit de l'image.
    """
    # Charger l'image en niveaux de gris
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Appliquer un filtre bilatéral pour réduire le bruit
    denoised = cv2.bilateralFilter(img, 9, 75, 75)

    # Sauvegarder l'image temporairement
    temp_path = "temp_denoised.jpg"
    cv2.imwrite(temp_path, denoised)
    return temp_path

def deskew_with_hough_transform(image_path, output_path):
    """
    Corriger l'inclinaison d'une image en utilisant la transformée de Hough.
    """
    # Charger l'image en niveaux de gris
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Appliquer un flou pour réduire le bruit
    blurred = cv2.GaussianBlur(img, (5, 5), 0)

    # Appliquer un seuil binaire inversé
    _, binary = cv2.threshold(blurred, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Trouver les contours pour mettre en évidence les lignes
    edges = cv2.Canny(binary, 50, 150, apertureSize=3)

    # Appliquer la transformée de Hough pour détecter les lignes
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

    # Calculer l'angle moyen des lignes détectées
    angles = []
    if lines is not None:
        for rho, theta in lines[:, 0]:
            angle = (theta * 180 / np.pi) - 90  # Convertir en degrés
            angles.append(angle)

    # Si aucune ligne n'est détectée, on retourne l'image d'origine
    if not angles:
        print("Aucune ligne détectée. L'image pourrait ne pas être inclinée.")
        cv2.imwrite(output_path, img)
        return output_path

    # Calculer l'angle moyen pour redresser l'image
    median_angle = np.median(angles)

    # Redresser l'image
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    # Sauvegarder l'image redressée
    cv2.imwrite(output_path, rotated)
    return output_path

def preprocess_image(image_path):
    """
    Pipeline complet de prétraitement d'une image :
    1. Améliorer le contraste et binariser.
    2. Supprimer le bruit.
    3. Corriger l'inclinaison avec Hough Transform.
    """
    # Étape 1 : Améliorer le contraste et binariser
    contrast_path = enhance_contrast_and_binarize(image_path)

    # Étape 2 : Suppression du bruit
    denoised_path = remove_noise(contrast_path)

    # Étape 3 : Correction de l'inclinaison avec Hough Transform
    directory, filename = os.path.split(image_path)
    preprocessed_filename = f"preprocessed_{filename}"
    preprocessed_path = os.path.join(directory, preprocessed_filename)
    final_path = deskew_with_hough_transform(denoised_path, preprocessed_path)

    # Nettoyer les fichiers temporaires
    if os.path.exists(contrast_path):
        os.remove(contrast_path)
    if os.path.exists(denoised_path):
        os.remove(denoised_path)

    return final_path

if __name__ == "__main__":
    # Ajouter un argument pour le chemin de l'image
    parser = argparse.ArgumentParser(description="Preprocess an image to improve OCR results.")
    parser.add_argument(
        "image_path",
        type=str,
        help="Path to the image file to preprocess."
    )
    args = parser.parse_args()

    # Exécuter le prétraitement
    preprocessed_path = preprocess_image(args.image_path)
    print(f"Image prétraitée sauvegardée ici : {preprocessed_path}")

