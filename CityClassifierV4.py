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

import pickle
import gzip

class CityClassifier():
    def __init__(self,show_results: True):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 'mps', 'cpu'
        print(device)
        self.device = device
        self.extractor0 = SIFT(max_num_keypoints=15000).eval().to(device)  # load the extractor for big map, more keypoints
        self.extractor1 = SIFT(max_num_keypoints=10000).eval().to(device)  # load the extractor for small region, less points
        self.matcher = LightGlue(features="sift").eval().to(device)
        self.show_results_enabled = show_results

    def show_result(self, query_image, map_img, kp_query, kp_map, good_matches):
        pass

    def classify(self, query_image_path):
        scores = {}
        device = self.device
        image1 = load_image(query_image_path).cuda()
        query_feats = self.extractor0.extract(image1.to(device))

        for subdir in tqdm(os.listdir("images/subimages")):
            subdir_path = os.path.join("images/subimages", subdir)
            if not os.path.isdir(subdir_path):
                continue
            for map_file in glob.glob(os.path.join(subdir_path, "*.jpg")):
                # print(f"Matching against {os.path.basename(map_file)}")
                t0 = time.time()
                # map_img = cv2.imread(map_file, cv2.IMREAD_GRAYSCALE)



                with gzip.open(map_file + '_glue_descriptors.bin.gz', 'rb') as file:
                    feats0 = pickle.load(file)

                feats1 = query_feats


                matches01 = self.matcher({"image0": feats0, "image1": feats1})
                feats0, feats1, matches01 = [
                    rbd(x) for x in [feats0, feats1, matches01]
                ]  # remove batch dimension

                kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
                m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]

                confidence = len(matches)

                scores[os.path.basename(map_file)] = confidence

                if self.show_results_enabled:
                    image0 = load_image(map_file).cuda()
                    axes = viz2d.plot_images([image0, image1])
                    viz2d.plot_matches(m_kpts0, m_kpts1, color="lime", lw=0.2)
                    viz2d.add_text(0, f'Stop after {matches01["stop"]} layers', fs=20)

                    # kpc0, kpc1 = viz2d.cm_prune(matches01["prune0"]), viz2d.cm_prune(matches01["prune1"])
                    # viz2d.plot_images([image0, image1])
                    # viz2d.plot_keypoints([kpts0, kpts1], colors=[kpc0, kpc1], ps=10)

                    # viz2d.save_plot(f"./results/{os.path.basename(query_image_path)[:-4]}/{os.path.basename(map_file)[:-4]}.png")
                    viz2d.save_plot(f"./results/{os.path.basename(map_file)[:-4]}.png")

                t1 = time.time()
        result = max(scores, key=scores.get)
        return result, scores[result]


    def generate_map_descriptors(self):
        device = self.device
        for subdir in tqdm(os.listdir("images/subimages")):
            subdir_path = os.path.join("images/subimages", subdir)
            if not os.path.isdir(subdir_path):
                continue
            for map_file in glob.glob(os.path.join(subdir_path, "*.jpg")):

                image0 = load_image(map_file).cuda()
                feats0 = self.extractor0.extract(image0.to(device))

                with gzip.open(map_file + '_glue_descriptors.bin.gz', 'wb') as file:
                    # Serialize and write the variable to the file
                    pickle.dump(feats0, file)

