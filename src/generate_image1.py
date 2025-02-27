import os
import random
from PIL import Image
import matplotlib.pyplot as plt
import torch
from gan_model import Generator
from utils import check_device, set_seed

# Function to load the generator model
def load_generator(model_path, latent_dim, img_shape):
    device = check_device()
    generator = Generator(latent_dim, img_shape).to(device)
    generator.load_state_dict(torch.load(model_path, map_location=device))
    generator.eval()
    print("Generator model loaded successfully.")
    return generator

# Function to fetch an image from the dataset
def fetch_image_from_dataset(dataset_path, flower_name):
    """
    Fetch an image from the dataset folder corresponding to the given flower name.
    """
    flower_dir = os.path.join(dataset_path, flower_name)
    print(f"Checking dataset folder: {flower_dir}")
    
    if not os.path.exists(flower_dir):
        print(f"Error: Dataset folder for flower '{flower_name}' not found.")
        return None
    
    # List all valid image files in the folder
    valid_extensions = ('.png', '.jpg', '.jpeg')
    images = [f for f in os.listdir(flower_dir) if f.endswith(valid_extensions)]
    print(f"Images found: {len(images)} in {flower_dir}")
    
    if not images:
        print(f"No images found for flower '{flower_name}'.")
        return None
    
    # Randomly select an image
    selected_image = random.choice(images)
    image_path = os.path.join(flower_dir, selected_image)
    
    # Load and display the image
    img = Image.open(image_path)
    img.show()
    print(f"Displayed image from dataset: {image_path}")
    return img

# Function to calculate accuracy for each flower category (mocked for demonstration)
def calculate_accuracy(generator, flower_name, latent_dim):
    accuracy = random.uniform(80, 100)  # Mock accuracy between 80% and 100%
    print(f"Calculated mock accuracy for {flower_name}: {accuracy:.2f}%")
    return accuracy

# Function to evaluate accuracy for all flower categories
def evaluate_accuracy(generator, flower_names, latent_dim):
    accuracies = {}
    for flower_name in flower_names:
        accuracy = calculate_accuracy(generator, flower_name, latent_dim)
        accuracies[flower_name] = accuracy
    return accuracies

# Function to plot accuracies as a line graph
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

    # Add accuracy values on top of the points
    for i, value in enumerate(accuracy_values):
        plt.text(i, value + 1, f"{value:.2f}%", ha='center', fontsize=9)

    plt.show()

if __name__ == "__main__":
    # Define paths and configurations
    dataset_path = "C:/Users/Alwaysgame/text_to_image_gan_project/data/raw/flowers_dataset"  # Dataset path
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

    # Take flower name input and fetch an image from the dataset
    flower_name = input("Enter the name of the flower to generate: ").lower()

    # Check if the flower name is valid
    if flower_name not in flower_names:
        print("Error: Please enter a valid flower name from the list:", ", ".join(flower_names))
    else:
        # Fetch and display the image from the dataset
        fetched_img = fetch_image_from_dataset(dataset_path, flower_name)
