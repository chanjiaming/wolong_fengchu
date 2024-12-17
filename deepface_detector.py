import cv2
import numpy as np
import os
folder_path = r'C:\Users\User\Desktop\hackathon\images' 

# Function to calculate Laplacian Variance (blurriness)
def calculate_laplacian_variance(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    blurred = cv2.GaussianBlur(image, (3, 3), 0)
    laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
    return laplacian.var()

# Function to calculate Edge Count using Canny Edge Detection
def count_edges(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    edges = cv2.Canny(image, 100, 200) 
    return np.count_nonzero(edges)

laplacian_variances = []
image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]


for filename in image_files:
    file_path = os.path.join(folder_path, filename)
    var = calculate_laplacian_variance(file_path)
    if var is not None:
        laplacian_variances.append(var)


dynamic_threshold = np.median(laplacian_variances) * 0.9
print(f"Dynamic Laplacian Variance Threshold: {dynamic_threshold:.2f}\n")


for filename in image_files:
    file_path = os.path.join(folder_path, filename)
    
    laplacian_var = calculate_laplacian_variance(file_path)
    edge_count = count_edges(file_path)

    if laplacian_var is None or edge_count is None:
        print(f"Error reading {filename}")
        continue

    if laplacian_var < dynamic_threshold and edge_count < 500:
        result = "blurry with low edges, possible deepfake"
    elif laplacian_var < dynamic_threshold:
        result = "blurry, possible deepfake"
    elif edge_count < 500:
        result = "low edges detected, suspicious"
    else:
        result = "sharp and likely real"

    print(f"{filename}: Laplacian Variance = {laplacian_var:.2f}, Edges = {edge_count}, Result: {result}")
