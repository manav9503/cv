import streamlit as st
import cv2
import numpy as np
from PIL import Image

def adjust_contrast(image, alpha=1.5):
    """
    Adjust contrast of the image.
    :param image: Input image
    :param alpha: Contrast control (1.0-3.0)
    :return: Contrast adjusted image
    """
    return cv2.convertScaleAbs(image, alpha=alpha)

def adjust_brightness(image, beta=50):
    """
    Adjust brightness of the image.
    :param image: Input image
    :param beta: Brightness control (0-100)
    :return: Brightness adjusted image
    """
    return cv2.convertScaleAbs(image, beta=beta)

def smoothen_image(image, kernel_size=(5, 5)):
    """
    Apply Gaussian Blur to smoothen the image.
    :param image: Input image
    :param kernel_size: Size of the Gaussian kernel
    :return: Smoothened image
    """
    return cv2.GaussianBlur(image, kernel_size, 0)

def sharpen_image(image):
    """
    Apply sharpening filter to the image.
    :param image: Input image
    :return: Sharpened image
    """
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)

def apply_mask(image):
    """
    Apply a binary black-and-white mask to highlight regions of interest and invert the mask.
    :param image: Input image
    :return: Inverse intensity masked image
    """
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.shape[2] == 3 else cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    
    # Create binary mask: areas of interest will be white, others black
    _, mask = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
    
    # Inverse the mask (flip black and white)
    inverted_mask = cv2.bitwise_not(mask)

    # Handle images with 3 channels (RGB) and 4 channels (RGBA)
    if image.shape[2] == 3:
        # For RGB image, we need to apply the mask directly to the 3 channels
        inverted_mask_rgb = cv2.cvtColor(inverted_mask, cv2.COLOR_GRAY2BGR)
        black_white_mask = np.zeros_like(image)
        black_white_mask[:, :] = inverted_mask_rgb  # Apply the 3-channel inverted mask
    elif image.shape[2] == 4:
        # For RGBA image, we need to ensure the mask has 4 channels (including alpha)
        inverted_mask_rgba = cv2.cvtColor(inverted_mask, cv2.COLOR_GRAY2BGRA)
        black_white_mask = np.zeros_like(image)
        black_white_mask[:, :] = inverted_mask_rgba  # Apply the 4-channel inverted mask
    
    # Return the inverse intensity masked image
    return black_white_mask

# Streamlit app
st.title("Image Enhancement App")
st.write("Upload an image and apply various enhancement techniques.")

# Enhancement options in the sidebar
st.sidebar.header("Enhancement Options")
contrast = st.sidebar.slider("Contrast", 1.0, 3.0, 1.5, 0.1)
brightness = st.sidebar.slider("Brightness", 0, 100, 50, 10)
smooth = st.sidebar.checkbox("Apply Smoothing")
sharpen = st.sidebar.checkbox("Apply Sharpening")
mask = st.sidebar.checkbox("Apply Masking")

# Image upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load the image
    image = np.array(Image.open(uploaded_file))
    st.image(image, caption="Original Image", use_container_width=True)

    # Apply enhancements
    enhanced_image = image.copy()

    # Adjust contrast
    enhanced_image = adjust_contrast(enhanced_image, alpha=contrast)

    # Adjust brightness
    enhanced_image = adjust_brightness(enhanced_image, beta=brightness)

    # Smoothing
    if smooth:
        enhanced_image = smoothen_image(enhanced_image)

    # Sharpening
    if sharpen:
        enhanced_image = sharpen_image(enhanced_image)

    # Masking
    if mask:
        enhanced_image = apply_mask(enhanced_image)

    # Display enhanced image
    st.image(enhanced_image, caption="Enhanced Image", use_container_width=True)
