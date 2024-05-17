import cv2
import numpy as np
import os
import glob
import time
from matplotlib import pyplot as plt
from tqdm import tqdm

"""
City Classifier
Experimental third version using LBP (Local Binary Patterns).
Improvements and limitations:
    - Works decently with scale
    - Works decently with rotation
    - Works with positioning
    - Is decently fast
    - Makes some mistakes
"""

class CityClassifier():
    def __init__(self, show_results=True):
        self.show_results_enabled = show_results

    def show_result(self, query_image, map_img, good_matches):
        img_merged = cv2.drawMatches(query_image, None, map_img, None, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        plt.imshow(img_merged)
        plt.title('Detected Point')
        plt.show()

    def classify(self, query_image_path):
        query_image = cv2.imread(query_image_path, cv2.IMREAD_GRAYSCALE)
        scores = {}

        for subdir in tqdm(os.listdir("images/subimages")):
            subdir_path = os.path.join("images/subimages", subdir)
            if not os.path.isdir(subdir_path):
                continue
            for map_file in glob.glob(os.path.join(subdir_path, "*.jpg")):
                t0 = time.time()
                map_img = cv2.imread(map_file, cv2.IMREAD_GRAYSCALE)

                lbp_query = cv2.calcHist([query_image], [0], None, [256], [0, 256])
                lbp_map = cv2.calcHist([map_img], [0], None, [256], [0, 256])

                similarity = cv2.compareHist(lbp_query, lbp_map, cv2.HISTCMP_CORREL)
                scores[os.path.basename(map_file)] = similarity

                t1 = time.time()
                if similarity > 0.9:  # Adjust this threshold as needed
                    if self.show_results_enabled:
                        self.show_result(query_image, map_img, None)

        return max(scores, key=scores.get)

    def generate_map_descriptors(self):
        """
        This function generates the descriptor points using LBP for all maps and saves them
        as a binary file so we don't have to compute them at classification time.

        """
        for subdir in os.listdir("images/subimages"):
            subdir_path = os.path.join("images/subimages", subdir)
            if not os.path.isdir(subdir_path):
                continue
            print(f"Generating descriptors for images in {subdir}")
            for map_file in glob.glob(os.path.join(subdir_path, "*.jpg")):
                print(f"Generating descriptors for {os.path.basename(map_file)}")
                t0 = time.time()
                map_img = cv2.imread(map_file, cv2.IMREAD_GRAYSCALE)

                lbp_hist = cv2.calcHist([map_img], [0], None, [256], [0, 256])
                # Open the file in binary mode
                with gzip.open(map_file + '_lbp_descriptors.bin.gz', 'wb') as file:
                    # Serialize and write the variable to the file
                    pickle.dump(lbp_hist, file)
