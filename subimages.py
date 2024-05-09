from PIL import Image
import os

def crop_image(input_image_path, output_directory, output_prefix):
    """
    Crop the input image into four subimages.

    Parameters:
    - input_image_path: Path to the input image.
    - output_directory: Path to the output directory to save cropped subimages.
    - output_prefix: Prefix to use for the output subimage filenames.
    """
    # Open the input image
    img = Image.open(input_image_path)
    
    # Get the dimensions of the input image
    width, height = img.size
    
    # Calculate the dimensions of each subimage
    sub_width = width // 2
    sub_height = height // 2
    
    # Create a subdirectory for the output images
    output_path = os.path.join(output_directory, output_prefix)
    os.makedirs(output_path, exist_ok=True)
    
    # Crop and save the four subimages
    for i in range(2):
        for j in range(2):
            # Calculate coordinates for cropping
            x = i * sub_width
            y = j * sub_height
            # Crop the image to get the subimage
            subimg = img.crop((x, y, x + sub_width, y + sub_height))
            # Save the subimage
            subimg.save(os.path.join(output_path, f"{output_prefix}_{i}{j}.jpg"))

# Function to crop all images in a directory
def crop_images_in_directory(input_directory, output_directory):
    """
    Crop all images in the input directory that end with .jpg.

    Parameters:
    - input_directory: Path to the input directory containing images.
    - output_directory: Path to the output directory to save cropped subimages.
    """
    # Iterate through files in the input directory
    for filename in os.listdir(input_directory):
        if filename.endswith(".jpg"):
            # Full path to the input image
            input_image_path = os.path.join(input_directory, filename)
            # Prefix for output subimage filenames
            output_prefix = os.path.splitext(filename)[0]
            # Crop the image and save the subimages in the output directory
            crop_image(input_image_path, output_directory, output_prefix)

# Example usage:
input_directory = "images"
output_directory = "images/subimages"
crop_images_in_directory(input_directory, output_directory)
