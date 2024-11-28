import cv2
import numpy as np
from PIL import Image, ImageEnhance
import os

class Preprocessor:
    @staticmethod
    def enhance_contrast_and_binarize(image_path):
        """
        Améliorer le contraste et binariser l'image.
        """
        img = Image.open(image_path)
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.5)  # Ajuster le contraste (facteur 1.5 pour éviter les écrasements)

        # Convertir en niveaux de gris
        img = img.convert("L")
        temp_path = "temp_gray.jpg"
        img.save(temp_path)

        # Charger avec OpenCV pour appliquer une binarisation adaptative
        img_cv = cv2.imread(temp_path, cv2.IMREAD_GRAYSCALE)
        adaptive_thresh = cv2.adaptiveThreshold(
            img_cv, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )

        # Sauvegarder l'image binarisée
        temp_path_binarized = "temp_binarized.jpg"
        cv2.imwrite(temp_path_binarized, adaptive_thresh)

        return temp_path_binarized

    @staticmethod
    def remove_noise(image_path):
        """
        Supprimer le bruit avec un filtre bilatéral.
        """
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # Filtrage bilatéral pour réduire le bruit tout en préservant les bords
        denoised = cv2.bilateralFilter(img, 5, 50, 50)

        # Sauvegarder l'image nettoyée
        temp_path = "temp_denoised.jpg"
        cv2.imwrite(temp_path, denoised)
        return temp_path

    @staticmethod
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

    @staticmethod
    def preprocess_image(image_path):
        """
        Pipeline complet de prétraitement.
        """
        contrast_path = Preprocessor.enhance_contrast_and_binarize(image_path)
        denoised_path = Preprocessor.remove_noise(contrast_path)
        directory, filename = os.path.split(image_path)
        preprocessed_filename = f"preprocessed_{filename}"
        preprocessed_path = os.path.join(directory, preprocessed_filename)
        final_path = Preprocessor.deskew_with_hough_transform(denoised_path, preprocessed_path)
        if os.path.exists(contrast_path):
            os.remove(contrast_path)
        if os.path.exists(denoised_path):
            os.remove(denoised_path)
        return final_path

