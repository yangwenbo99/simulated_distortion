# Import necessary libraries
import cv2
import kornia
import torch
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread("lenna.png")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Convert the image to a tensor
image_tensor = kornia.image_to_tensor(image, keepdim=False).float() / 255.0

# Define the motion blur parameters
kernel_size = 11
angle = 45  # Direction of the motion blur

# Apply the motion blur
motion_blur1 = kornia.filters.MotionBlur(kernel_size, angle, direction=1.0)
blurred_image_tensor1 = motion_blur1(image_tensor)

# Convert the tensor back to an image
blurred_image1 = kornia.tensor_to_image(blurred_image_tensor1 * 255.0).astype('uint8')
print(blurred_image1.shape, blurred_image1.dtype, blurred_image1.max())


motion_blur2 = kornia.filters.MotionBlur(kernel_size, angle, direction=0)
# Apply the motion blur to create blurred_image2
blurred_image_tensor2 = motion_blur2(image_tensor)

# Convert the tensor back to an image
blurred_image2 = kornia.tensor_to_image(blurred_image_tensor2 * 255.0).astype('uint8')

# Display the original, blurred_image1, and blurred_image2
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(image)
plt.title("Original Image")
plt.subplot(1, 3, 2)
plt.imshow(blurred_image1)
plt.title("Motion Blurred Image 1")
plt.subplot(1, 3, 3)
plt.imshow(blurred_image2)
plt.title("Motion Blurred Image 2")
plt.show()