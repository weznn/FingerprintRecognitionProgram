
pip
install
opencv - python
numpy
scikit - image

import cv2
import numpy as np
from skimage.morphology import skeletonize
from skimage.filters import threshold_otsu


def preprocess_fingerprint(image_path):
    """
    Parmak izi görüntüsünü işleyip skeletonize yapar.
    """
    # Görüntüyü gri tonlamaya dönüştür
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Görsel yüklenemedi. Dosya yolunu kontrol edin.")

    # Görüntüyü eşikleme ile ikili (binary) hale getir
    thresh = threshold_otsu(image)
    binary_image = image > thresh

    # Skeletonize işlemi (parmak izini inceltme)
    skeleton = skeletonize(binary_image)
    return skeleton


def match_fingerprints(img1, img2):
    """
    İki parmak izi arasında benzerlik karşılaştırması yapar.
    """
    # Görüntüler arasındaki farkları hesapla
    difference = np.sum(img1 != img2)
    similarity = 1 - (difference / img1.size)
    return similarity


if __name__ == "__main__":
    # Test için iki parmak izi görseli kullan
    fingerprint1_path = "fingerprint1.jpg"
    fingerprint2_path = "fingerprint2.jpg"

    try:
        print("Parmak izleri işleniyor...")
        fingerprint1 = preprocess_fingerprint(fingerprint1_path)
        fingerprint2 = preprocess_fingerprint(fingerprint2_path)

        # Benzerlik oranını hesapla
        similarity_score = match_fingerprints(fingerprint1, fingerprint2)
        print(f"Benzerlik oranı: {similarity_score * 100:.2f}%")

        if similarity_score > 0.8:
            print("Parmak izleri eşleşiyor.")
        else:
            print("Parmak izleri eşleşmiyor.")
    except Exception as e:
        print(f"Hata: {e}")

