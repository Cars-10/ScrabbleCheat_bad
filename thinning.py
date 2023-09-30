import cv2
import numpy as np

def thinning(image_path):
    # Read the image and convert to grayscale
    img = cv2.imread(image_path, 0)

    # Binarize the image: You may need to adjust the threshold value
    _, bin_img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)

    # Define the structure element for morphological operations
    struct_elem = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))

    # Iteratively apply thinning
    thinning_img = np.copy(bin_img)  # Make a copy to not modify the original
    prev_img = np.zeros(bin_img.shape, np.uint8)
    while cv2.countNonZero(thinning_img - prev_img) != 0:
        prev_img = thinning_img.copy()
        eroded = cv2.erode(thinning_img, struct_elem)
        temp_img = cv2.dilate(eroded, struct_elem)
        thinning_img = cv2.subtract(thinning_img, temp_img)

    # Invert the image back to original form
    thinning_img = cv2.bitwise_not(thinning_img)

    # Display
    cv2.imshow("Original", img)
    cv2.imshow("Thinning", thinning_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Call the function
thinning('original_board_image.jpg')
