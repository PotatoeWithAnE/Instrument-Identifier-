import os
import tensorflow as tf
import cv2

"""
#check for images that are not working
for category in os.listdir(data_dir):
    print("====================",category,"checking ====================")
    image_list = os.listdir(os.path.join(data_dir,category))

    for img in image_list:
        img_dir = os.path.join(data_dir, category, img)
        img_mat = cv2.imread(img_dir)

        if img_mat is None:
            print(f"Error reading image: {img_dir}")
        else:
            cv2.waitKey(1)
            cv2.destroyAllWindows()
"""


# Set data_dir to the directory containing your images
data_dir = 'data_'+input("training or testing? (\"train\" or \"test)\"")


# Function to check image format
def check_image_format(image_path):
    try:
        with open(image_path, 'rb') as img_file:
            img_data = img_file.read()
            tf.image.decode_image(img_data)  # Attempt to decode the image
        return True  # Image format is supported
    except tf.errors.InvalidArgumentError as e:
        print(f"Invalid image format: {image_path} ({e})")
        return False  # Image format is not supported


# Iterate through images and check for format issues
for category in os.listdir(data_dir):
    category_dir = os.path.join(data_dir, category)
    for img in os.listdir(category_dir):
        img_path = os.path.join(category_dir, img)

        if not check_image_format(img_path):
            # Print the path of the problematic image
            print(f"Problematic image: {img_path}")
        else:
            print(img_path)
