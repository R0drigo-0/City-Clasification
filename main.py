import cv2

# from CityClassifierV1 import CityClassifier
from CityClassifierV2 import CityClassifier

c = CityClassifier(show_results = False)
query = cv2.imread("./images/train/Berlin_crop_13.png", cv2.IMREAD_GRAYSCALE)
result = c.classify(query)
print(f"Result: {result}")
exit()


