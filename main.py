import glob

import cv2

import csv

from CityClassifierV1 import CityClassifier
# from CityClassifierV2 import CityClassifier
# from CityClassifierV3 import CityClassifier
from CityClassifierV4 import CityClassifier

c = CityClassifier(show_results = True)
c.generate_map_descriptors()

# result = c.classify(query)
# print(f"Result: {result}")

# Test all images in test

test_images = glob.glob("./images/test/*.png")
# test_images = ["./images/test/Barcelona_crop_11.png"]
# test_images = ["./images/test-extended/maps06.png"]
with open('./results/V4.csv', 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(["Input file", "Prediction"])
    for query in test_images:
        # query = cv2.imread(im, cv2.IMREAD_GRAYSCALE)
        result = c.classify(query)
        csvwriter.writerow([query, result])
        print(f"Result for {query}: {result}")



exit()


