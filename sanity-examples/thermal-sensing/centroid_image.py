import numpy as np
import cv2
from scipy.ndimage import center_of_mass
import matplotlib.pyplot as plt

from PIL import Image

def calculate_heading_angle(centroid, image_shape, camera_fov_y):
    """
    Calculate the heading angle to turn the robot towards the centroid in the image.
    
    Parameters:
    - centroid: (cx, cy) - the centroid coordinates of the hot area.
    - image_shape: (height, width) - the dimensions of the image (height, width).
    - camera_fov_x: float - the horizontal field of view of the camera in degrees.
    
    Returns:
    - heading_angle: float - the angle (in degrees) the robot needs to turn to face the centroid.
    """

    # NOTE REMEMBER THAT X IS UP DOWN AND Y IS SIDE IN CAMERA SPACE
    
    # Get the center of the image (cx_center, cy_center)
    cx_center, cy_center = image_shape[1] // 2, image_shape[0] // 2  # image shape: (height, width)
    
    # Get the displacement from the center to the centroid
    delta_y = centroid[1] - cy_center  # horizontal displacement in pixels
    
    # Calculate the angle per pixel
    angle_per_pixel = camera_fov_y / image_shape[1]  # field of view in degrees per pixel
    
    # Calculate the heading angle in degrees (relative to the center of the image)
    heading_angle = delta_y * angle_per_pixel
    
    return heading_angle

def find_centroid_of_hot_area(temperature_array, threshold):
    """
    Finds the centroid of the hot area in a temperature array.
    
    Parameters:
    - temperature_array (np.ndarray): 2D array representing the temperature values.
    - threshold (float): The temperature threshold above which a point is considered "hot".
    
    Returns:
    - tuple: (y, x) coordinates of the centroid of the hot area.
    """
    
    # Step 1: Create a binary mask for the hot areas (areas above the threshold)
    hot_mask = temperature_array > threshold
    
    # Step 2: Find the centroid using the center_of_mass function from scipy
    centroid = center_of_mass(hot_mask)
    
    # Step 3: Return the centroid coordinates
    return centroid

# Function to load the PNG image as a NumPy array
def load_image_as_array(image_path):
    # Open the image using Pillow
    image = Image.open(image_path)
    
    # Convert the image to a NumPy array
    image_array = np.array(image)
    
    return image_array



# Show the first few pixel values
if __name__ == "__main__":
    # Example: A 2D array representing the temperature values

        # Example usage:
    image_path = './thermal-test-0/thermal-data-027.png'  # Provide the path to your PNG image
    image = Image.open(image_path)

    image_array = load_image_as_array(image_path)

    # Print the shape and type of the array for verification
    print(f"Image shape: {image_array.shape}")
    print(f"Image dtype: {image_array.dtype}")
    print(np.array(image_array))


    # temperature_array = np.array([[20, 25, 30, 35, 40],
    #                              [20, 25, 30, 35, 45],
    #                              [20, 25, 30, 40, 50],
    #                              [20, 25, 35, 45, 55],
    #                              [20, 25, 30, 40, 60]])

    # Define a threshold temperature value to identify "hot areas"
    threshold_temperature = 40
    
    # Find the centroid of the hot area
    centroid = find_centroid_of_hot_area(image_array, threshold_temperature)
    
    print(f"The centroid of the hot area is at: {centroid}")
    plt.imshow(image_array)
    # plt.axis('off')  # Optionally remove axis for a cleaner look
    plt.show()

    image_shape = image.size  # (width, height)
    camera_fov_y = 110  # Horizontal FOV of the camera in degrees (adjust for your camera)

    # Calculate the heading angle
    heading_angle = calculate_heading_angle(centroid, image_shape, camera_fov_y)

    # Output the result
    print(f"The robot needs to turn {heading_angle:.2f} degrees to face the centroid.")

