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
from matplotlib import pyplot as plt
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

def my_track_points(gray, num_features):
    corners = cv.goodFeaturesToTrack(gray, num_features, 0.01, 10)
    return np.int0(corners)

def my_point_rotation(gray, point, patch_size=16):
    x, y = int(point[0]), int(point[1])
    half = patch_size // 2
    
    # Extract patch around point
    y1, y2 = max(y - half, 0), min(y + half, gray.shape[0])
    x1, x2 = max(x - half, 0), min(x + half, gray.shape[1])
    patch = gray[y1:y2, x1:x2]

    # Compute gradients
    grad_x = cv.Sobel(patch, cv.CV_64F, 1, 0, ksize=3)
    grad_y = cv.Sobel(patch, cv.CV_64F, 0, 1, ksize=3)

    # Magnitude and orientation
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    orientation = np.arctan2(grad_y, grad_x) * 180 / np.pi  # in degrees
    orientation = (orientation + 360) % 360  # normalize to [0, 360)

    # Weighted histogram of orientations
    hist, bin_edges = np.histogram(
        orientation, bins=36, range=(0, 360), weights=magnitude
    )

    # Dominant angle
    dominant_angle = bin_edges[np.argmax(hist)]
    
    return dominant_angle
    

def my_descriptor(gray, points, patch_size=40, small_size=8):
    descriptors = []
    w,h = gray.shape

    for point in points:
        x,y = int(point[0]), int(point[1])
        half = patch_size // 2

        angle = my_point_rotation(gray, (x,y), half)

        M = cv.getRotationMatrix2D((x,y), -angle, 1.0)
        rotated = cv.warpAffine(gray, M, (w, h), flags=cv.INTER_LINEAR)

        half = patch_size // 2
        y1, y2 = max(y - half, 0), min(y + half, h)
        x1, x2 = max(x - half, 0), min(x + half, w)
        patch = rotated[y1:y2, x1:x2]
        
        # skip points too close to the border
        if patch.shape[0] < patch_size // 2 or patch.shape[1] < patch_size // 2:
            continue

        patch = patch.astype(np.float32)
        patch -= np.mean(patch)
        std = np.std(patch)
        if std > 1e-5:
            patch /= std

        small_patch = cv.resize(patch, (small_size, small_size), interpolation=cv.INTER_AREA)
        small_patch -= small_patch.min()
        if small_patch.max() > 0:
            small_patch /= small_patch.max()
        descriptor = small_patch.flatten()

        descriptors.append(descriptor)

    return np.array(descriptors, dtype=np.float32)


def my_distance(desc1, desc2):
    return np.linalg.norm(desc1 - desc2)


def my_match(descs1, descs2, ratio_threshold=0.75):
    good_matches = []

    for i, d1 in enumerate(descs1):
        # Compute all distances between d1 and descs2
        distances = np.linalg.norm(descs2 - d1, axis=1)
        if len(distances) < 2:
            continue
        sorted_idx = np.argsort(distances)
        best, second_best = distances[sorted_idx[0]], distances[sorted_idx[1]]

        if best / second_best < ratio_threshold:
            j = sorted_idx[0]
            good_matches.append((i, j, best))

    return good_matches



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
        sum_histograms = np.zeros((3, 256), dtype=np.float64)
        hist_r, hist_g, hist_b = create_histograms(img)
        sum_histograms[0] += hist_r[:,0]
        sum_histograms[1] += hist_g[:,0]
        sum_histograms[2] += hist_b[:,0]

        for j in range(i + 1, len(images)):
            image2 = images[j]
            
            img2 = cv.imread(f"{image_dir}/{image2}")
            img2 = cv.cvtColor(img2, cv.COLOR_BGR2RGB)
            img2 = resize_image(img2)
            dist = histogram_distance(img, img2)
            if dist > dist_treshold:
                break
            similar.append(j)
            hist_r, hist_g, hist_b = create_histograms(img2)
            sum_histograms[0] += hist_r[:,0]
            sum_histograms[1] += hist_g[:,0]
            sum_histograms[2] += hist_b[:,0]

        
        # create folder with similar images   
        if len(similar) > 1:
            folder_name = f"similar-{similar_count}"
            folder_path = os.path.join(output_dir, folder_name)
            os.makedirs(folder_path, exist_ok=True)

            # Compute average histogram and average intensities
            count_sim = len(similar) + 1
            avg_r = sum_histograms[0] / count_sim
            avg_g = sum_histograms[1] / count_sim
            avg_b = sum_histograms[2] / count_sim
            bins = np.arange(256)

            mean_r = np.sum(bins * avg_r) / np.sum(avg_r)
            mean_g = np.sum(bins * avg_g) / np.sum(avg_g)
            mean_b = np.sum(bins * avg_b) / np.sum(avg_b)
            target_mean = (mean_r + mean_g + mean_b) / 3.0

            scale = np.array([target_mean / mean_r,
                              target_mean / mean_g,
                              target_mean / mean_b])

            # print(f"[{folder_name}] White balance scale: R={scale[0]:.2f}, G={scale[1]:.2f}, B={scale[2]:.2f}")

            # Save and white-balance all grouped images directly
            group_indices = [i] + similar
            for idx in group_indices:
                img_path = os.path.join(image_dir, images[idx])
                img_orig = cv.imread(img_path)
                img_rgb = cv.cvtColor(img_orig, cv.COLOR_BGR2RGB).astype(np.float32)

                # Apply white balance scale
                img_balanced = img_rgb * scale[np.newaxis, np.newaxis, :]
                img_balanced = np.clip(img_balanced, 0, 255).astype(np.uint8)

                out_name = images[idx]
                # cv.imwrite(os.path.join(folder_path, out_name), cv.cvtColor(img_rgb, cv.COLOR_RGB2BGR))
                # cv.imwrite(os.path.join(folder_path, f"balanced_{out_name}"), cv.cvtColor(img_balanced, cv.COLOR_RGB2BGR))
                cv.imwrite(os.path.join(folder_path, out_name), cv.cvtColor(img_balanced, cv.COLOR_RGB2BGR))

            # Save average histogram plot
            plt.figure(figsize=(6, 3))
            plt.bar(bins, avg_r, color='red', alpha=0.4, label='Red')
            plt.bar(bins, avg_g, color='green', alpha=0.4, label='Green')
            plt.bar(bins, avg_b, color='blue', alpha=0.4, label='Blue')
            plt.title("RGB Histogram (Average)")
            plt.xlabel("Pixel Value")
            plt.ylabel("Frequency")
            plt.legend()
            plt.savefig(os.path.join(folder_path, "histograms.jpg"), bbox_inches="tight", dpi=300)
            plt.close()

            similar_count += 1


            