import cv2

# from CityClassifierV1 import CityClassifier
from CityClassifierV2 import CityClassifier

c = CityClassifier(show_results = True)
query = cv2.imread("./images/train/Barcelona_crop_0.png", cv2.IMREAD_GRAYSCALE)
result = c.classify(query)
print(f"Result: {result}")
exit()


