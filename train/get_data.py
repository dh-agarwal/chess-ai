import cv2
import numpy as np
import os

def load_and_warp(image_path, src_points, dst_size=750, expansion_factor=1.08):
    """
    Create two perspective transforms - normal and expanded
    Args:
        image_path: Path to the image
        src_points: Corner points of the board
        dst_size: Size of the output warped image
        expansion_factor: How much to expand the border region (1.2 = 20% expansion)
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image at {image_path}")

    dst_points = np.array([
        [0, 0],
        [dst_size, 0],
        [dst_size, dst_size],
        [0, dst_size]
    ], dtype="float32")
    transform_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    warped = cv2.warpPerspective(image, transform_matrix, (dst_size, dst_size))
    
    # Calculate expanded source points by moving each corner outward
    center = np.mean(src_points, axis=0)
    expanded_src_points = np.array([
        # Top corners: expand outward and upward
        src_points[0] + np.array([(src_points[0][0] - center[0]) * (expansion_factor - 1), 
                                (src_points[0][1] - center[1]) * (expansion_factor - 1)]),
        src_points[1] + np.array([(src_points[1][0] - center[0]) * (expansion_factor - 1), 
                                (src_points[1][1] - center[1]) * (expansion_factor - 1)]),
        # Bottom corners: expand outward and downward
        src_points[2] + np.array([(src_points[2][0] - center[0]) * (expansion_factor - 1),
                                (src_points[2][1] - center[1]) * (expansion_factor - 1)]),
        src_points[3] + np.array([(src_points[3][0] - center[0]) * (expansion_factor - 1),
                                (src_points[3][1] - center[1]) * (expansion_factor - 1)])
    ], dtype="float32")
    
    # Create expanded warped image (same destination size as regular)
    expanded_transform = cv2.getPerspectiveTransform(expanded_src_points, dst_points)
    expanded_warped = cv2.warpPerspective(image, expanded_transform, (dst_size, dst_size))
    
    return warped, expanded_warped

def divide_into_squares(warped_image, expanded_warped, board_size=8):
    """Extract squares from both warped images, using expanded version for border squares.
    Applies more padding upward and less padding sideways.
    
    Args:
        warped_image: The regular perspective-corrected board image
        expanded_warped: The expanded perspective-corrected board image
        board_size: Number of squares per side (default 8 for chess)
    """
    h, w = warped_image.shape[:2]
    square_h, square_w = h // board_size, w // board_size
    
    # Calculate different padding sizes for vertical and horizontal
    regular_vertical_padding = int(square_h * 0.25)    # 50% padding upward
    downward_padding = int(square_h * 0.15)           # 15% padding downward
    regular_horizontal_padding = int(square_w * 0.3)  # 20% padding sideways
    border_vertical_padding = int(square_h * 0.25)    # 35% padding upward for border
    border_horizontal_padding = int(square_w * 0.25)  # 15% padding sideways for border
    
    squares = []
    for i in range(board_size):
        row = []
        for j in range(board_size):
            is_border = (i == 0 or i == board_size-1 or j == 0 or j == board_size-1)
            
            y1 = i * square_h
            x1 = j * square_w
            y2 = (i + 1) * square_h
            x2 = (j + 1) * square_w
            
            source_img = expanded_warped if is_border else warped_image
            vert_padding = border_vertical_padding if is_border else regular_vertical_padding
            horiz_padding = border_horizontal_padding if is_border else regular_horizontal_padding
            
            # Apply paddings
            padded_x1 = max(0, x1 - horiz_padding)
            padded_x2 = min(w, x2 + horiz_padding)
            padded_y1 = max(0, y1 - vert_padding)
            padded_y2 = min(h, y2 + downward_padding)  # Add downward padding
            
            square = source_img[padded_y1:padded_y2, padded_x1:padded_x2]
            row.append(square)
            
        squares.append(row)
    return squares

def main():
    # Make sure output directory exists
    os.makedirs("temp_squares", exist_ok=True)
    
    # Example: Adjust src_points to match your board setup
    src_points = np.array([
        [255, 970],
        [3952, 1063],
        [3946, 4688],
        [230, 4758]
    ], dtype="float32")

    # Gather all images
    image_directory = "train_images"
    image_files = sorted([f for f in os.listdir(image_directory) if f.endswith('.png')])
    ind = 0
    for i, filename in enumerate(image_files, start=1):
        print(f"Processing image {i}/{len(image_files)}: {filename}")
        image_path = os.path.join(image_directory, filename)
            
        warped_image, expanded_warped = load_and_warp(image_path, src_points)
        squares = divide_into_squares(warped_image, expanded_warped)
        print(len(squares))
        
        squares_images_paths = []
        
        for row_idx, row in enumerate(squares):
            row_paths = []
            for col_idx, square in enumerate(row):
                square_path = f"temp_squares/square{ind}.jpg"
                ind += 1
                cv2.imwrite(square_path, square)
                row_paths.append(square_path)
            squares_images_paths.append(row_paths)

if __name__ == "__main__":
    main()