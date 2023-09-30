import cv2
import pytesseract

# Load image and convert to grayscale
img = cv2.imread('scrabble_board.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Crop the image to the desired region

x, y, w, h =  0, 500, 960, 960
board   = gray[y:y+h, x:x+w]

gray = board
# Save the grayscale image as a new file
cv2.imwrite('board_image.jpg', board)

# Threshold to get binary image
_, thresh = cv2.threshold(gray, 190, 255, cv2.THRESH_BINARY_INV)
cv2.imwrite('thresh_board_image.jpg', thresh)

# Find contours to identify each cell
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(contours)
# Assuming each square is about the same size, find average contour area and filter out small noise
avg_area = sum(cv2.contourArea(contour) for contour in contours) / len(contours)
squares = [contour for contour in contours if 0.8 < cv2.contourArea(contour)/avg_area < 1.2]

# For each square, extract the letter using Tesseract
letters = []
for square in squares:

    print(square)
    x, y, w, h = cv2.boundingRect(square)
    roi = gray[y:y+h, x:x+w]
    letter = pytesseract.image_to_string(roi, config='--psm 10').strip()
    letters.append((x, y, letter))

# Sort squares by position and formulate words. (This part is a bit tricky and may require additional refinement.)
letters.sort(key=lambda x: (x[1], x[0]))  # Sort by y first, then by x
# ... Extract words based on Scrabble rules ...

print(letters)
