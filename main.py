import cv2
# import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import glob
import time
import os

class CityClassifier():
    def __init__(self):
        pass

    def show_result(self, map_img, top_left, bottom_right, res):
        img_merged = cv2.merge([map_img, map_img, map_img])
        cv2.rectangle(img_merged, top_left, bottom_right, (0, 0, 255), 10)

        plt.subplot(121), plt.imshow(res, cmap='gray')
        plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
        plt.subplot(122), plt.imshow(img_merged)
        # plt.subplot(122), plt.imshow(img, cmap='gray')
        plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
        # plt.suptitle(method)

        plt.show()

    def classify(self, query_image):
        scores = {}
        for map_file in glob.glob("./images/*.jpg"):
            print(f"Matching against {os.path.basename(map_file)}")
            t0 = time.time()
            map_img = cv2.imread(map_file, cv2.IMREAD_GRAYSCALE)

            w, h = query_image.shape[::-1]

            method = cv2.TM_CCOEFF

            # Apply template Matching
            res = cv2.matchTemplate(map_img, query_image, method)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

            # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
            if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                top_left = min_loc
                confidence = min_val
            else:
                top_left = max_loc
                confidence = max_val

            # print(confidence)
            scores[os.path.basename(map_file)] = confidence

            bottom_right = (top_left[0] + w, top_left[1] + h)

            t1 = time.time()
            print(f"Done in {t1-t0} seconds.")

        # print(sorted(scores.items(), key=lambda x:x[1]))
        return max(scores, key=scores.get)




c = CityClassifier()
query = cv2.imread("./images/train/Barcelona_crop_0.png", cv2.IMREAD_GRAYSCALE)
result = c.classify(query)
print(f"Result: {result}")
exit()


