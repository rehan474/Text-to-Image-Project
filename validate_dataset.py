from PIL import Image
import os

folder_path = 'C:/Users/Alwaysgame/text_to_image_gan_project/data/processed/train_data'

for root, _, files in os.walk(folder_path):
    for file in files:
        file_path = os.path.join(root, file)
        try:
            img = Image.open(file_path)
            img.verify()  # Check if it's a valid image file
        except (IOError, SyntaxError):
            print(f"Removing corrupt file: {file_path}")
            os.remove(file_path)
