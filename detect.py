import cv2
import numpy as np

def load_and_warp(image_path, src_points, dst_size=750):
    image = cv2.imread(image_path)
    dst_points = np.array([
        [0, 0],
        [dst_size, 0],
        [dst_size, dst_size],
        [0, dst_size]
    ], dtype="float32")
    transform_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    warped = cv2.warpPerspective(image, transform_matrix, (dst_size, dst_size))
    return warped

def divide_into_squares(warped_image, board_size=8):
    h, w = warped_image.shape[:2]
    square_h, square_w = h // board_size, w // board_size
    squares = []
    for i in range(board_size):
        row = []
        for j in range(board_size):
            square = warped_image[i * square_h:(i + 1) * square_h, j * square_w:(j + 1) * square_w]
            row.append(square)
        squares.append(row)
    return squares

def calculate_square_descriptor(square):
    return cv2.mean(square)[0]

def map_to_chess_notation(row, col, left_white=True):
    files = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    if not left_white:
        files.reverse()
    return f"{files[col]}{8 - row}"

src_points = np.array([
    [952, 438],
    [2978, 430],
    [3455, 2524],
    [625, 2605]
], dtype="float32")  # Update these points as needed

base_image = load_and_warp('game_images/1.jpg', src_points)
base_squares = divide_into_squares(base_image)
base_descriptors = [[calculate_square_descriptor(square) for square in row] for row in base_squares]

cv2.imshow("Warped Base Image", base_image)
cv2.waitKey(0)

left_white = None

for i in range(2, 7):
    current_image = load_and_warp(f'game_images/{i}.jpg', src_points)
    cv2.imshow("Warped Image", current_image)
    cv2.waitKey(0)

    current_squares = divide_into_squares(current_image)
    current_descriptors = [[calculate_square_descriptor(square) for square in row] for row in current_squares]

    change_diffs = []
    for row in range(8):
        for col in range(8):
            diff = abs(base_descriptors[row][col] - current_descriptors[row][col])
            change_diffs.append(((row, col), diff))
    
    change_diffs.sort(key=lambda x: x[1], reverse=True)
    top_changes = change_diffs[:2]

    print(f"Top 2 changes in game_images/{i}.jpg:")
    for change in top_changes:
        square_coord, intensity_change = change
        row, col = square_coord
        if left_white is None:
            left_white = col < 4
            white_side = "left" if left_white else "right"
            print(f"Determined that the {white_side} half is white.")

        chess_notation = map_to_chess_notation(row, col, left_white)
        print(f"Square {chess_notation} (Grid: {square_coord}) with intensity change of {intensity_change}")

        square_image = current_squares[row][col]
        cv2.imshow(f"Changed Square {chess_notation}", square_image)
        cv2.waitKey(0)  # Show the changed square; press any key to close it and proceed to the next

    base_descriptors = current_descriptors

cv2.destroyAllWindows()
