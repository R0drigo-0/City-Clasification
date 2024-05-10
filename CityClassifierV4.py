"""
City Classifier
Improved version with Deep Learning
https://github.com/cvg/LightGlue
- Same as V2 but probably faster and more accurate
"""
import os

from lightglue import LightGlue, SuperPoint, DISK, SIFT, ALIKED, DoGHardNet
from lightglue.utils import load_image, rbd
from lightglue import viz2d
import torch

import cv2
import numpy as np
import os
import glob
import time
from matplotlib import pyplot as plt

from tqdm import tqdm

class CityClassifier():
    def __init__(self,show_results: True):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 'mps', 'cpu'
        self.device = device
        self.extractor0 = SIFT(max_num_keypoints=30000).eval().to(device)  # load the extractor for big map, more keypoints
        self.extractor1 = SIFT(max_num_keypoints=5000).eval().to(device)  # load the extractor for small region, less points
        self.matcher = LightGlue(features="sift").eval().to(device)
        self.show_results_enabled = show_results

    def show_result(self, query_image, map_img, kp_query, kp_map, good_matches):
        pass

    def classify(self, query_image):
        scores = {}
        device = self.device

        for map_file in tqdm(glob.glob("images/*.jpg")):
            # print(f"Matching against {os.path.basename(map_file)}")
            t0 = time.time()
            # map_img = cv2.imread(map_file, cv2.IMREAD_GRAYSCALE)

            image0 = load_image(map_file).cuda()
            image1 = load_image(query_image).cuda()


            feats0 = self.extractor0.extract(image0.to(device), resize=None)
            feats1 = self.extractor1.extract(image1.to(device), resize=None)
            matches01 = self.matcher({"image0": feats0, "image1": feats1})
            feats0, feats1, matches01 = [
                rbd(x) for x in [feats0, feats1, matches01]
            ]  # remove batch dimension

            kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
            m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]

            confidence = len(matches)

            scores[os.path.basename(map_file)] = confidence

            t1 = time.time()
            # print(f"Done in {t1-t0} seconds. Confidence {confidence}")

            # if len(good_matches) > 10:  # Adjust this threshold as needed
            #     if self.show_results_enabled:
            #         self.show_result(query_image, map_img, kp_query, _, good_matches)
        result = max(scores, key=scores.get)
        return result, scores[result]