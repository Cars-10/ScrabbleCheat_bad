import cv2
import numpy as np

def preprocess_image(image_path):
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise and help contour detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Use Canny edge detection
    edged = cv2.Canny(blurred, 50, 150)

    return image, edged

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
    # Define points for a 600x600 pixel image, this value can be changed
    dest = np.array([
        [0, 0],
        [599, 0],
        [599, 599],
        [0, 599]
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
    warped = cv2.warpPerspective(image, matrix, (600, 600))

    return warped

def segment_board_into_squares(warped_image):
    squares = []
    square_size = warped_image.shape[0] // 15  # Assuming a standard 15x15 Scrabble board

    for i in range(15):
        for j in range(15):
            top_left_y = i * square_size
            bottom_right_y = (i + 1) * square_size
            top_left_x = j * square_size
            bottom_right_x = (j + 1) * square_size

            square = warped_image[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
            squares.append(square)

    return squares

# Example usage:
image_path = "scrabble_board.jpg"
original_image, preprocessed_image = preprocess_image(image_path)
board_contour = find_board(preprocessed_image)

if board_contour is not None:
    warped = warp_board(original_image, board_contour)
    squares = segment_board_into_squares(warped)

    for index, square in enumerate(squares):
        cv2.imshow(f"Square {index + 1}", square)
        cv2.waitKey(0)

    cv2.destroyAllWindows()
else:
    print("Board not found!")
