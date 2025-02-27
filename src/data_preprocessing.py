import os
from PIL import Image
import torch
from torchvision import transforms

def preprocess_images(input_dir, output_dir, image_size=(64, 64)):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor()
    ])

    for flower_type in os.listdir(input_dir):
        flower_path = os.path.join(input_dir, flower_type)
        output_path = os.path.join(output_dir, flower_type)

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        for image_name in os.listdir(flower_path):
            img_path = os.path.join(flower_path, image_name)
            img = Image.open(img_path).convert("RGB")
            img_tensor = transform(img)

            output_file_path = os.path.join(output_path, image_name)
            torch.save(img_tensor, output_file_path)
            print(f"Processed and saved {output_file_path}")

if __name__ == "__main__":
    input_dir = "C:/Users/Alwaysgame/text_to_image_gan_project/data/raw/flowers_dataset"
    output_dir = "C:/Users/Alwaysgame/text_to_image_gan_project/data/processed/train_data"
    preprocess_images(input_dir, output_dir)
