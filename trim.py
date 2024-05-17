import os
import random
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

def is_mostly_black(image):
    gray_image = image.convert('L')
    average_intensity = sum(gray_image.getdata()) / (image.width * image.height)
    threshold = 60
    return average_intensity < threshold

def trim_crop_rotate_images(input_folder, output_folder, crop_count=100, crop_size=(256, 256)):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for filename in os.listdir(input_folder):
        if filename.endswith('.jpg'):
            input_path = os.path.join(input_folder, filename)
            output_path_prefix = os.path.join(output_folder, os.path.splitext(filename)[0])
            with Image.open(input_path) as img:
                img = img.crop(img.getbbox())
                for i in range(crop_count):
                    x = random.randint(0, img.width - crop_size[0])
                    y = random.randint(0, img.height - crop_size[1])
                    angle = random.randint(0, 360)
                    crop = img.rotate(angle).crop((x, y, x + crop_size[0], y + crop_size[1]))
                    # Check if the crop is mostly black, and try again if it is
                    if is_mostly_black(crop):
                        continue
                    crop.save(f"{output_path_prefix}_crop_{i}.png")

input_folder = "images"
output_folder = "images/test"
trim_crop_rotate_images(input_folder, output_folder)
