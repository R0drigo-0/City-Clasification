import cv2
# import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import glob
import time
import os

"""
City Classifier
Second version using SIFT (Scale Invariant Feature Transform).
Improvements and limitations:
    - Works with scale
    - Works with rotation
    - Works with positioning
    - Is slower than V1 (27s per map vs 0.9s)
"""
import cv2
import numpy as np
import os
import glob
import time
from matplotlib import pyplot as plt

class CityClassifier():
    def __init__(self,show_results: False):
        self.sift = cv2.SIFT_create()
        self.show_results_enabled = show_results

    def show_result(self, query_image, map_img, kp_query, kp_map, good_matches):
        img_merged = cv2.drawMatches(query_image, kp_query, map_img, kp_map, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        plt.imshow(img_merged)
        plt.title('Detected Point')
        plt.show()

    def classify(self, query_image):
        scores = {}
        kp_query, des_query = self.sift.detectAndCompute(query_image, None)

        for map_file in glob.glob("./images/*.jpg"):
            print(f"Matching against {os.path.basename(map_file)}")
            t0 = time.time()
            map_img = cv2.imread(map_file, cv2.IMREAD_GRAYSCALE)

            kp_map, des_map = self.sift.detectAndCompute(map_img, None)

            # FLANN parameters
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)

            flann = cv2.FlannBasedMatcher(index_params, search_params)

            # Matching descriptor vectors using KNN
            matches = flann.knnMatch(des_query, des_map, k=2)

            # Ratio test as per Lowe's paper
            good_matches = []
            for m, n in matches:
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)

            confidence = len(good_matches)

            scores[os.path.basename(map_file)] = confidence

            t1 = time.time()
            print(f"Done in {t1-t0} seconds. Confidence {confidence}")

            if len(good_matches) > 10:  # Adjust this threshold as needed
                # self.show_result(map_img, None, None, kp_query, kp_map, good_matches)
                if self.show_results_enabled:
                    self.show_result(query_image, map_img, kp_query, kp_map, good_matches)

        return max(scores, key=scores.get)