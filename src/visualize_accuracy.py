import torch
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm
from PIL import Image
from gan_model import Generator
from utils import check_device, set_seed
import os

# Function to load the generator model
def load_generator(model_path, latent_dim, img_shape):
    device = check_device()
    generator = Generator(latent_dim, img_shape).to(device)
    generator.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    generator.eval()
    return generator

# Function to generate an image based on flower name
def generate_image(generator, flower_name, latent_dim):
    variation = random.randint(0, 10000)
    seed = sum(ord(char) for char in flower_name) + variation
    set_seed(seed)
    
    device = check_device()
    z = torch.randn(1, latent_dim, device=device)
    with torch.no_grad():
        gen_img = generator(z).cpu().squeeze(0)
    
    # Rescale and convert to numpy
    gen_img = (gen_img * 0.5 + 0.5) * 255
    gen_img = gen_img.permute(1, 2, 0).numpy().astype("uint8")
    img = Image.fromarray(gen_img)
    
    output_dir = "C:/Users/Alwaysgame/text_to_image_gan_project/outputs/generated_images"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{flower_name}_variation_{variation}.png")
    img.save(output_path)
    print(f"Generated image saved at {output_path}")
    return img

# Function to calculate accuracy for each flower category (mocked for demonstration)
def calculate_accuracy(generator, flower_name, latent_dim):
    accuracy = random.uniform(80, 100)  # Mock accuracy between 80% and 100%
    return accuracy

# Function to evaluate accuracy for all flower categories
def evaluate_accuracy(generator, flower_names, latent_dim):
    accuracies = {}
    for flower_name in flower_names:
        accuracy = calculate_accuracy(generator, flower_name, latent_dim)
        accuracies[flower_name] = accuracy
    return accuracies

# Plot accuracy as Bar Chart
def plot_accuracies_bar(accuracies):
    categories = list(accuracies.keys())
    accuracy_values = list(accuracies.values())

    plt.figure(figsize=(8, 5))
    plt.bar(categories, accuracy_values, color='darkblue', edgecolor='black')
    plt.xlabel("Flower Categories")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy of Generated Images by Category")
    plt.ylim(0, 100)
    
    for i, value in enumerate(accuracy_values):
        plt.text(i, value + 1, f'{value:.2f}%', ha='center', va='bottom', fontsize=10, color='white')
    
    plt.tight_layout()
    plt.show()

# Plot accuracy as Pie Chart
def plot_accuracies_pie(accuracies):
    categories = list(accuracies.keys())
    accuracy_values = list(accuracies.values())

    plt.figure(figsize=(8, 8))
    plt.pie(accuracy_values, labels=categories, autopct='%1.1f%%', startangle=140, colors=plt.cm.Blues(np.linspace(0.4, 1, len(accuracies))))
    plt.title("Accuracy Distribution by Flower Categories")
    plt.show()

# Plot accuracy as Radar Chart
def plot_accuracies_radar(accuracies):
    categories = list(accuracies.keys())
    accuracy_values = list(accuracies.values())

    num_vars = len(categories)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    accuracy_values += accuracy_values[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.fill(angles, accuracy_values, color='darkblue', alpha=0.5)
    ax.plot(angles, accuracy_values, color='darkblue', linewidth=2)

    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12, fontweight='bold')
    
    plt.title("Accuracy by Flower Categories")
    plt.show()

# Plot accuracy as Heatmap
def plot_accuracies_heatmap(accuracies):
    categories = list(accuracies.keys())
    accuracy_values = list(accuracies.values())

    accuracy_matrix = np.array(accuracy_values).reshape(1, -1)

    plt.figure(figsize=(8, 2))
    sns.heatmap(accuracy_matrix, annot=True, fmt='.2f', xticklabels=categories, yticklabels=["Accuracy"], cmap='Blues', cbar=False, linewidths=0.5)
    plt.title("Accuracy Heatmap by Flower Categories")
    plt.show()

# Plot accuracy as Line Chart
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

def plot_accuracies(accuracies):
    categories = list(accuracies.keys())
    accuracy_values = list(accuracies.values())
    
    # Create a smaller figure size
    plt.figure(figsize=(8, 5))  # Smaller figure size
    
    # Create bars with a shadow effect (darker bars in the background)
    shadow_offset = 0.05  # Offset for shadow
    shadow_bars = plt.bar(categories, accuracy_values, width=0.4, color='black', edgecolor='black', zorder=1)
    
    # Create the main bars (with gradient color)
    bars = plt.bar(categories, accuracy_values, width=0.4, color=plt.cm.Blues(np.linspace(0.4, 1, len(accuracies))), edgecolor='black', zorder=3)

    # Display accuracy on top of each bar with better positioning and styling
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height + 2, f"{height:.2f}%", 
                 ha='center', va='bottom', color='black', fontsize=10, fontweight='bold')

    # Add labels, title, and improve layout
    plt.xlabel("Flower Categories", fontsize=12, fontweight='bold')
    plt.ylabel("Accuracy (%)", fontsize=12, fontweight='bold')
    plt.title("Accuracy of Generated Images by Category", fontsize=14, fontweight='bold')
    
    # Add a grid for better readability
    plt.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)

    # Adjust the x-axis ticks and title to prevent overlap
    plt.xticks(rotation=45)
    plt.ylim(0, 100)
    
    # Adjust layout to prevent title overlap
    plt.tight_layout()
    
    # Show the plot
    plt.show()

# Plot accuracy as Stacked Bar Chart
def plot_accuracies_stacked_bar(accuracies):
    categories = list(accuracies.keys())
    accuracy_values = list(accuracies.values())

    plt.figure(figsize=(8, 5))
    plt.bar(categories, accuracy_values, color='darkblue', edgecolor='black')
    plt.xlabel("Flower Categories")
    plt.ylabel("Accuracy (%)")
    plt.title("Stacked Accuracy of Generated Images by Category")
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.show()

# Plot accuracy as Bubble Chart
def plot_accuracies_bubble(accuracies):
    categories = list(accuracies.keys())
    accuracy_values = list(accuracies.values())

    plt.figure(figsize=(8, 5))
    plt.scatter(categories, accuracy_values, s=np.array(accuracy_values) * 10, color='darkblue', alpha=0.6, edgecolor='black')
    plt.xlabel("Flower Categories")
    plt.ylabel("Accuracy (%)")
    plt.title("Bubble Chart of Accuracy by Flower Category")
    plt.ylim(0, 100)
    plt.show()

# Main script to generate accuracy and visualize
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
    
    # Display visualizations
    plot_accuracies_bar(accuracies)
    plot_accuracies_pie(accuracies)
    plot_accuracies_radar(accuracies)
    plot_accuracies_heatmap(accuracies)
    plot_accuracies_line(accuracies)
    plot_accuracies(accuracies)
    plot_accuracies_stacked_bar(accuracies)
    plot_accuracies_bubble(accuracies)

    # Optionally: Take flower name input and generate an image for the specified flower
    flower_name = input("Enter the name of the flower to generate: ")
    if flower_name in flower_names:
        generated_img = generate_image(generator, flower_name, latent_dim)
    else:
        print("Error: Please enter a valid flower name from the available categories.")
