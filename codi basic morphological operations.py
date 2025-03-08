import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import thin

# Load the grayscale image
image = cv2.imread('test.jpg', cv2.IMREAD_GRAYSCALE)

# Display the original image
plt.figure(figsize=(5, 5))
plt.imshow(image, cmap='gray')
plt.title("Original Image")
plt.axis("off")
plt.show()

# Convert to binary image (ensure background is black and text is white)
_, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)

# Define a larger kernel to enhance erosion
kernel = np.ones((4, 4), np.uint8)  # Increased from (3,3) to (4,4)

# Apply morphological transformations
dilated = cv2.dilate(binary, kernel, iterations=1)
eroded = cv2.erode(binary, kernel, iterations=1)  # This should work correctly now

# Other transformations
gradient = cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, kernel)
internal_boundary = cv2.subtract(binary, eroded)
external_boundary = cv2.subtract(dilated, binary)
thinning = thin(binary // 255) * 255
thickening = cv2.dilate(binary, kernel, iterations=1)

# Fast skeletonization with OpenCV
binary_skel = binary.copy()
skel = np.zeros(binary_skel.shape, np.uint8)
element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))

while True:
    eroded_temp = cv2.erode(binary_skel, element)
    temp = cv2.dilate(eroded_temp, element)
    temp = cv2.subtract(binary_skel, temp)
    skel = cv2.bitwise_or(skel, temp)
    binary_skel = eroded_temp.copy()
    if cv2.countNonZero(binary_skel) == 0:
        break

# Display all images in a single plot
titles = ["Original", "Dilated", "Eroded", "Gradient", "Internal Boundary", 
          "External Boundary", "Thinning", "Thickening", "Skeletonization"]
images = [binary, dilated, eroded, gradient, internal_boundary, 
          external_boundary, thinning.astype(np.uint8), thickening, skel]

plt.figure(figsize=(10,10))
for i in range(len(images)):
    plt.subplot(3, 3, i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis("off")

plt.tight_layout()
plt.show()
