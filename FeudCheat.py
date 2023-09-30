# %%
import cv2
import numpy as np
import pytesseract
import matplotlib.pyplot as plt



from ImageConverter import *

def preprocess_image(image_path, x, y, w, h):
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)[y:y+h, x:x+w]
    clean_image =  process_scrabble_image2(image, debug=True)
    return image, clean_image

def find_board(image):
    # Find contours in the image
    contours, _ = cv2.findContours(image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours by area in descending order
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    # Look for the large square (Scrabble board)
    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

        # If our approximated contour has four points, it's likely the board
        if len(approx) == 4:
            return approx

    return None

def warp_board(image, board_contour):
    # The `warp_board` function takes an image and a board contour as input, and warps the image to a
    # 960x960 pixel size based on the perspective transform defined by the board contour.

    # :param image: The input image that you want to warp. It should be a 600x600 pixel image
    # :param board_contour: The parameter "board_contour" is a numpy array that represents the contour of
    # the board in the image. It should contain the coordinates of the four corners of the board in a
    # specific order. The order of the points should be such that the top-left corner is the first point,
    # followed by the
    # :return: the warped image after applying a perspective transform.

    # Define points for a 960x960 pixel image, this value can be changed
    dest = np.array([
        [0, 0],
        [959, 0],
        [959, 959],
        [0, 959]
    ], dtype='float32')


    # Order points in the contour in a consistent way
    board_contour = board_contour.reshape(4, 2)
    ordered_points = np.zeros((4, 2), dtype='float32')


    # Sort points based on their x-coordinates
    sorted_points = sorted(board_contour, key=lambda pt: pt[0])

    # Leftmost point is top-left, rightmost is top-right
    ordered_points[0] = sorted_points[0]
    ordered_points[2] = sorted_points[1]

    # Sort the leftmost and rightmost points by their y-coordinates to get the top-left/bottom-left and top-right/bottom-right
    if sorted_points[0][1] < sorted_points[1][1]:
        ordered_points[0], ordered_points[2] = sorted_points[0], sorted_points[1]
    else:
        ordered_points[0], ordered_points[2] = sorted_points[1], sorted_points[0]

    # The remaining points are the top-right and bottom-right
    if sorted_points[2][1] < sorted_points[3][1]:
        ordered_points[1], ordered_points[3] = sorted_points[2], sorted_points[3]
    else:
        ordered_points[1], ordered_points[3] = sorted_points[3], sorted_points[2]

    # Compute the perspective transform matrix and warp the image
    matrix = cv2.getPerspectiveTransform(ordered_points, dest)
    warped = cv2.warpPerspective(image, matrix, (960, 960))

    return warped

def segment_board_into_squares(warped_image):
    squares = []
    square_size = warped_image.shape[0] // 15  # Assuming a standard 15x15 Scrabble board
    print(square_size)
    for i in range(15):
        for j in range(15):
            top_left_y = i * square_size
            bottom_right_y = (i + 1) * square_size
            top_left_x = j * square_size
            bottom_right_x = (j + 1) * square_size

            square = warped_image[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
            squares.append(square)

    return squares


debug = False
# Example usage:
image_path = "scrabble_board.jpg"
original_image, preprocessed_image = preprocess_image(image_path,0, 500, 960, 960)
board_contour = find_board(preprocessed_image)

if debug:
    cv2.imwrite('preprocessed_board_image.jpg', preprocessed_image)
    cv2.imwrite('original_board_image.jpg', original_image)


if board_contour is not None:
    #warped = warp_board(original_image, board_contour)
    squares = segment_board_into_squares(preprocessed_image)

count = 0
for square in squares:
    count += 1
    # Extract the tile from the square
    # Adjust the coordinates as needed
    gray = square[16:56, 12:52]

    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    thresh = cv2.threshold(blur, 0, 255,  cv2.THRESH_OTSU)[1]
    height, width = thresh.shape[:2]
    mask = np.zeros((height+2, width+2), np.uint8)  # +2 is necessary for the mask
    cv2.floodFill(thresh, mask, (0,0), 0)
    cv2.floodFill(thresh, mask, (15,39), 0) # X, R
    cv2.floodFill(thresh, mask, (15,15), 0) #D
    cv2.floodFill(thresh, mask, (width-1,height-1), 0)
    filled_letters = cv2.bitwise_not(thresh)

    filled_letters = process_scrabble_image3(filled_letters, debug=True)

    text = pytesseract.image_to_string(filled_letters, config='--psm 10 --user-words tile.words')

    if text != "":
        cv2.imwrite(f'thresh_tile_image_{count}.jpg', thresh)
        print(f'Tile {count}: {text}', end="")

    if count == 20 :
        exit()

    print("Board not found!")
    break



# %%
