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
    return squares, square_h, square_w

def calculate_square_descriptor(square):
    return cv2.mean(square)[0]

def map_to_chess_notation(row, col, left_white=True):
    ranks = ['1', '2', '3', '4', '5', '6', '7', '8']
    files = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    if not left_white:
        files.reverse()
        ranks.reverse()
    return f"{files[col]}{ranks[row]}"

# Initial setup for the base image
src_points = np.array([
    [952, 438],
    [2978, 430],
    [3455, 2524],
    [625, 2605]
], dtype="float32")  # Update these points as needed

base_image = load_and_warp('game_images/1.jpg', src_points)
base_squares, square_h, square_w = divide_into_squares(base_image)
base_descriptors = [[calculate_square_descriptor(square) for square in row] for row in base_squares]

cv2.imshow("Warped Base Image", base_image)
cv2.waitKey(0)

left_white = None

# Process each subsequent image to detect moves
for i in range(2, 7):  # Adjust the range as per your number of images
    current_image = load_and_warp(f'game_images/{i}.jpg', src_points)
    current_squares, square_h, square_w = divide_into_squares(current_image)
    current_descriptors = [[calculate_square_descriptor(square) for square in row] for row in current_squares]

    # Calculate intensity differences to detect changes
    change_diffs = []
    for row in range(8):
        for col in range(8):
            diff = abs(base_descriptors[row][col] - current_descriptors[row][col])
            change_diffs.append(((row, col), diff))
    
    # Sort and pick the top 2 changes
    change_diffs.sort(key=lambda x: x[1], reverse=True)
    top_changes = change_diffs[:2]
    print(f"Top 2 changes in game_images/{i}.jpg:")

    # Coordinates for "moved from" and "moved to"
    moved_from, moved_to = top_changes[0][0], top_changes[1][0]
    
    for square_coord, intensity_change in top_changes:
        row, col = square_coord
        if left_white is None:
            left_white = col < 4
            white_side = "left" if left_white else "right"
            print(f"Determined that the {white_side} half is white.")

        chess_notation = map_to_chess_notation(row, col, left_white)
        print(f"Square {chess_notation} (Grid: {square_coord}) with intensity change of {intensity_change}")

        # Draw rectangle overlay on detected squares
        top_left = (col * square_w, row * square_h)
        bottom_right = ((col + 1) * square_w, (row + 1) * square_h)
        color = (0, 255, 0) if square_coord == moved_to else (0, 0, 255)
        cv2.rectangle(current_image, top_left, bottom_right, color, 2)

        # Display chess notation
        notation_position = (col * square_w + 5, (row + 1) * square_h - 5)
        cv2.putText(current_image, chess_notation, notation_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Draw an arrow from "moved_from" to "moved_to"
    from_center = (moved_from[1] * square_w + square_w // 2, moved_from[0] * square_h + square_h // 2)
    to_center = (moved_to[1] * square_w + square_w // 2, moved_to[0] * square_h + square_h // 2)
    cv2.arrowedLine(current_image, from_center, to_center, (255, 0, 0), 2, tipLength=0.3)

    # Overlay grid and labels
    for row in range(8):
        for col in range(8):
            # Draw grid lines
            cv2.line(current_image, (col * square_w, 0), (col * square_w, 8 * square_h), (0, 255, 0), 1)
            cv2.line(current_image, (0, row * square_h), (8 * square_w, row * square_h), (0, 255, 0), 1)
            
            # Display notation for each square
            chess_notation = map_to_chess_notation(row, col, left_white)
            notation_position = (col * square_w + 5, (row + 1) * square_h - 5)
            cv2.putText(current_image, chess_notation, notation_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Display the annotated current image
    cv2.imshow("Warped Image with Move Overlay and Grid", current_image)
    cv2.waitKey(0)

    # Update the base descriptors for the next iteration
    base_descriptors = current_descriptors

cv2.destroyAllWindows()