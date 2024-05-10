import glob

import cv2

# from CityClassifierV1 import CityClassifier
# from CityClassifierV2 import CityClassifier
from CityClassifierV4 import CityClassifier

c = CityClassifier(show_results = False)
# c.generate_map_descriptors()
query = cv2.imread("./images/train/Barcelona_crop_1.png", cv2.IMREAD_GRAYSCALE)
query = "./images/train/Barcelona_crop_1.png"

# result = c.classify(query)
# print(f"Result: {result}")

# Test all images in test
test_images = glob.glob("./images/train/*.png")
for query in test_images:
    # query = cv2.imread(im, cv2.IMREAD_GRAYSCALE)
    result = c.classify(query)
    print(f"Result for {query}: {result}")



exit()


