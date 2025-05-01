import pytesseract
import cv2
from PIL import Image
import numpy as np

def read_image_stats(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray, lang='fra')  # ou 'eng' selon la langue

    return text