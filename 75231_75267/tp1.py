###############################################################################
## Computer Vision 2025-2026 - NOVA FCT
## Assignment 1
##
## Student 1
## Student 2
## 
###############################################################################

import cv2 as cv
import numpy as np
import os
import glob
import shutil
import re

image_dir = "./input"
output_dir = "./output"
dist_treshold = 0.15

def resize_image(image, size=512):
    w, h = image.shape[:2]
    if h < w:
        new_h = size
        new_w = int(w * (size / h))
    else:
        new_w = size
        new_h = int(h * (size / w))
    
    resized = cv.resize(image, (new_w, new_h), interpolation=cv.INTER_AREA)
    return resized

def create_histograms(img):
    hist_r = cv.calcHist([img], [0], None, [256], [0, 256])
    hist_g = cv.calcHist([img], [1], None, [256], [0, 256])
    hist_b = cv.calcHist([img], [2], None, [256], [0, 256])
    cv.normalize(hist_r, hist_r)
    cv.normalize(hist_g, hist_g)
    cv.normalize(hist_b, hist_b)
    
    return hist_r, hist_g, hist_b

def histogram_distance(img1, img2):
    hist_r1, hist_g1, hist_b1 = create_histograms(img1)
    hist_r2, hist_g2, hist_b2 = create_histograms(img2)
    dist_r = cv.compareHist(hist_r1, hist_r2, cv.HISTCMP_BHATTACHARYYA)
    dist_g = cv.compareHist(hist_g1, hist_g2, cv.HISTCMP_BHATTACHARYYA)
    dist_b = cv.compareHist(hist_b1, hist_b2, cv.HISTCMP_BHATTACHARYYA)
    avg_dist = (dist_r + dist_g + dist_b) / 3
    return avg_dist


if __name__ == "__main__":
    print("Assignment 1")

    # empty the output folder
    for filename in os.listdir(output_dir):
        file_path = os.path.join(output_dir, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")

    # images = sorted(glob.glob(os.path.join(image_dir, ".")))

    # Get all image files (jpg, png, jpeg) 
    image_paths = sorted(glob.glob(os.path.join(image_dir, "*.*")))
    
        
    images = sorted(
        [image for image in os.listdir(image_dir) if image.endswith(".jpg")],
        key=lambda x: int(re.search(r'\d+', x).group())
    )
    
    similar_count = 0
    for i in range(len(images) - 1):
        image = images[i]
        
        img = cv.imread(f"{image_dir}/{image}")
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img = resize_image(img)
        
        #find similar images
        similar = []
        for j in range(i + 1, len(images)):
            image2 = images[j]
            
            img2 = cv.imread(f"{image_dir}/{image2}")
            img2 = cv.cvtColor(img2, cv.COLOR_BGR2RGB)
            img2 = resize_image(img2)
            dist = histogram_distance(img, img2)
            if dist > dist_treshold:
                break
            similar.append(j)
        
        # create folder with similar images   
        if len(similar) > 1:
            folder_name = f"similar-{similar_count}"
            folder_path = os.path.join(output_dir, folder_name)
            os.makedirs(folder_path, exist_ok=True)
            shutil.copy(os.path.join(image_dir, image), os.path.join(folder_path, image))
            for idx in similar:
                similar_image = images[idx]
                shutil.copy(os.path.join(image_dir, similar_image), os.path.join(folder_path, similar_image))
            similar_count += 1
            