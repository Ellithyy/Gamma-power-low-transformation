# Import necessary libraries
import cv2
import numpy as np
import os
import random
import matplotlib.pyplot as plt
from google.colab import drive
import glob

# Step 1: Mount Google Drive (if your images are in Google Drive)
drive.mount('/content/drive')

# Step 2: Set the folder path where your images are stored
image_folder = '/content/drive/MyDrive/bonetest'  # Change this path to your image folder

# Step 3: Get a list of all image file paths from the folder
image_paths = glob.glob(os.path.join(image_folder, '*.jpg'))  # You can adjust the file extension if necessary

# Step 4: Randomly select 10 images from the folder
selected_images = random.sample(image_paths, 10)

# Function to apply Power-law (Gamma) transformation
def power_law_transformation(image, gamma=1.2, c=1):
    # Normalize image to range [0, 1]
    normalized_image = image / 255.0
    # Apply the power-law transformation
    transformed_image = c * np.power(normalized_image, gamma)
    # Scale back to [0, 255]
    return np.uint8(transformed_image * 255)

# Function to calculate RMSE
def calculate_rmse(original, processed):
    return np.sqrt(np.mean((original - processed) ** 2))

# Function to calculate PSNR
def calculate_psnr(original, processed):
    mse = np.mean((original - processed) ** 2)
    if mse == 0:  # Avoid division by zero
        return float('inf')
    max_pixel = 255.0
    return 20 * np.log10(max_pixel / np.sqrt(mse))

# Step 5: Loop through each selected image, apply power-law transformation, and display the results
plt.figure(figsize=(15, 15))

sample_size = len(selected_images)

for i, image_path in enumerate(selected_images):
    # Read the image
    img = cv2.imread(image_path)

    # Convert from BGR (OpenCV default) to RGB for displaying in Matplotlib
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Apply Power-law transformation
    gamma_img = power_law_transformation(img, gamma=1.2)  # Adjust gamma as needed

    gamma_img_rgb = cv2.cvtColor(gamma_img, cv2.COLOR_BGR2RGB)

    # Calculate RMSE and PSNR
    rmse = calculate_rmse(img, gamma_img)
    psnr = calculate_psnr(img, gamma_img)

    # Plot original image
    plt.subplot(sample_size, 2, 2 * i + 1)
    plt.imshow(img_rgb)
    plt.title(f"Original Image {i+1}")
    plt.axis('off')

    # Plot gamma-transformed image with annotations
    plt.subplot(sample_size, 2, 2 * i + 2)
    plt.imshow(gamma_img_rgb)
    plt.title(f"Gamma Image {i+1}\nRMSE: {rmse:.2f}, PSNR: {psnr:.2f} dB")
    plt.axis('off')

# Display the results
plt.tight_layout()
plt.show()