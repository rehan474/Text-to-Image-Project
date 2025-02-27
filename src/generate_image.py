import torch
import random
from gan_model import Generator
from utils import check_device, set_seed
from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np

# Function to load the generator model
def load_generator(model_path, latent_dim, img_shape):
    device = check_device()
    generator = Generator(latent_dim, img_shape).to(device)
    generator.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    generator.eval()
    return generator

# Function to generate an image based on flower name
def generate_image(generator, flower_name, latent_dim):
    # Generate a random variation
    variation = random.randint(0, 10000)
    
    # Create a unique seed using flower name and random variation
    seed = sum(ord(char) for char in flower_name) + variation
    set_seed(seed)
    
    device = check_device()
    z = torch.randn(1, latent_dim, device=device)
    with torch.no_grad():
        gen_img = generator(z).cpu().squeeze(0)
    
    # Rescale image to [0, 255] and convert to numpy
    gen_img = (gen_img * 0.5 + 0.5) * 255
    gen_img = gen_img.permute(1, 2, 0).numpy().astype("uint8")
    img = Image.fromarray(gen_img)
    
    # Display the generated image
    img.show()
    
    # Save the generated image with variation number
    output_dir = "C:/Users/Alwaysgame/text_to_image_gan_project/outputs/generated_images"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{flower_name}_variation_{variation}.png")
    img.save(output_path)
    print(f"Generated image saved at {output_path}")
    return img

# Function to calculate accuracy for each flower category (mocked for demonstration)
def calculate_accuracy(generator, flower_name, latent_dim):
    # Here you would have actual evaluation logic using a classifier or similarity check
    # This example uses a mock accuracy value for demonstration purposes
    accuracy = random.uniform(80, 100)  # Mock accuracy between 80% and 100%
    return accuracy

# Function to evaluate accuracy for all flower categories
def evaluate_accuracy(generator, flower_names, latent_dim):
    accuracies = {}
    for flower_name in flower_names:
        accuracy = calculate_accuracy(generator, flower_name, latent_dim)
        accuracies[flower_name] = accuracy
    return accuracies

def plot_accuracies_line(accuracies):
    categories = list(accuracies.keys())
    accuracy_values = list(accuracies.values())

    plt.figure(figsize=(8, 5))
    plt.plot(categories, accuracy_values, marker='o', color='darkblue', linestyle='-', linewidth=2)
    plt.xlabel("Flower Categories")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy Over Categories")
    plt.ylim(0, 100)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()



if __name__ == "__main__":
    model_path = "C:/Users/Alwaysgame/text_to_image_gan_project/models/gan_trained.pth"
    latent_dim = 128
    img_shape = 128
    flower_names = ["rose", "jasmine", "hibiscus", "periwinkle", "crossandra"]
    
    # Load the generator model
    generator = load_generator(model_path, latent_dim, img_shape)
    
    # Evaluate the model and get accuracy for each flower category
    accuracies = evaluate_accuracy(generator, flower_names, latent_dim)
    print("Accuracy per category:", accuracies)
    
    # Plot the accuracy graph
    plot_accuracies_line(accuracies)
    
    # Take flower name input and generate image for the specified flower
    flower_name = input("Enter the name of the flower to generate: ").lower()
    
    # Check if the flower name is valid
    if flower_name not in flower_names:
        print("Error: Please enter a valid flower name from the list:", ", ".join(flower_names))
    else:
        # Generate image for the valid flower name
        generated_img = generate_image(generator, flower_name, latent_dim)
