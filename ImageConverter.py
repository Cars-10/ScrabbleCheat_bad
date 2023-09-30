import cv2
import pytesseract
import matplotlib.pyplot as plt


def process_scrabble_image1(image, debug=False):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise and help contour detection
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    if debug:
        cv2.imwrite('blurred_board_image.jpg', blurred)

    # Use Canny edge detection
    edged = cv2.bitwise_not(cv2.Canny(blurred, 50, 150))
    return edged

def process_scrabble_image2(image, debug=False):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    thresh = cv2.threshold(blurred, 0, 255,  cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Increase contrast
    contrast_image = cv2.equalizeHist(thresh)
    if debug:
        cv2.imwrite('contrast_image.jpg', contrast_image)

    # Binarize the image
    _, binary_image = cv2.threshold(contrast_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if debug:
        cv2.imwrite('binary_image.jpg', binary_image)

    # Noise reduction
    denoised_image = cv2.bitwise_not(cv2.medianBlur(binary_image, 3))
    if debug:
        cv2.imwrite('denoised_image.jpg', denoised_image)

    # Use Canny edge detection
    edged = cv2.bitwise_not(cv2.Canny(denoised_image, 50, 150))
    if debug:
        cv2.imwrite('edged_image.jpg', edged)

    return edged




def process_scrabble_image3(image, debug=False):
    for i in range(3):
        eroded = cv2.erode(image.copy(), None, iterations=i + 1)
        # load image using cv2....and do processing.
        #plt.imshow(f"Eroded {i + 1} times", eroded)
        plt.imshow(eroded)
        # as opencv loads in BGR format by default, we want to show it in RGB.
        plt.show()
    return eroded
