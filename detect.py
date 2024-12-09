import cv2
import numpy as np
import os
import requests
import base64
import concurrent.futures
import time
import tkinter as tk
from tkinter import messagebox
import random
import ollama

PIECE_IMAGES = {}

def load_and_warp(image_path, src_points, dst_size=750, expansion_factor=1.2):
    """
    Warp the image with two perspectives - normal and expanded for border regions
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
    regular_vertical_padding = int(square_h * 0.15)    # 50% padding upward
    regular_horizontal_padding = int(square_w * 0.2)  # 20% padding sideways
    border_vertical_padding = int(square_h * 0.15)    # 35% padding upward for border
    border_horizontal_padding = int(square_w * 0.15)  # 15% padding sideways for border
    
    squares = []
    for i in range(board_size):
        row = []
        for j in range(board_size):
            # Determine if this is a border square
            is_border = (i == 0 or i == board_size-1 or j == 0 or j == board_size-1)
            
            # Calculate base coordinates for this square
            y1 = i * square_h
            x1 = j * square_w
            y2 = (i + 1) * square_h
            x2 = (j + 1) * square_w
            
            # Select the appropriate image and padding
            source_img = expanded_warped if is_border else warped_image
            vert_padding = border_vertical_padding if is_border else regular_vertical_padding
            horiz_padding = border_horizontal_padding if is_border else regular_horizontal_padding
            
            # Apply different padding amounts for vertical and horizontal
            padded_x1 = max(0, x1 - horiz_padding)  # Less left padding
            padded_x2 = min(w, x2 + horiz_padding)  # Less right padding
            padded_y1 = max(0, y1 - vert_padding)   # More top padding
            padded_y2 = y2                          # No bottom padding
            
            # Extract square from appropriate source
            square = source_img[padded_y1:padded_y2, padded_x1:padded_x2]
            row.append(square)
            
        squares.append(row)
    return squares
def detect_piece(img_path):
    """Use Ollama to detect chess pieces in an image"""
    prompt = """This is a zoomed in image of a square on a chess board. If the square is empty, respond with the word 'None'. 
    If there is a piece on the square, respond with the color of the piece, followed by a space, followed by the name of the piece. 
    For example, your responses may look like:
    'white rook'
    'black bishop'
    'white king'
    Ensure that your responses do not have any additional content - do not include full sentences, periods, or otherwise extraneous information. 
    Additionally, your responses should be in all lowercase. Your responses should NOT look like:
    'This square is empty'
    'Black pawn'
    'There is nothing on this square'"""
    
    # response = ollama.chat(
    #     model='llama3.2-vision:11b',
    #     messages=[{
    #         'role': 'user',
    #         'content': prompt,
    #         'images': [img_path]
    #     }]
    # )
    pieces = [
        'white king', 'white queen', 'white rook', 'white bishop', 'white knight', 'white pawn',
        'black king', 'black queen', 'black rook', 'black bishop', 'black knight', 'black pawn',
        'none'  
    ]

    selected_piece = random.choice(pieces)
    return selected_piece

    # return response.message.content.strip().lower()

def piece_to_fen_symbol(piece_str):
    """Convert piece description to FEN symbol"""
    if piece_str == "none":
        return ""
        
    mapping = {
        "white king": "K",
        "white queen": "Q",
        "white rook": "R",
        "white bishop": "B",
        "white knight": "N",
        "white pawn": "P",
        "black king": "k",
        "black queen": "q",
        "black rook": "r",
        "black bishop": "b",
        "black knight": "n",
        "black pawn": "p"
    }
    return mapping.get(piece_str, "")

def generate_fen_from_predictions(squares_images_paths):
    fen_ranks = []
    for row in range(8):
        rank_pieces = []
        for col in range(8):
            piece_str = detect_piece(squares_images_paths[row][col])
            symbol = piece_to_fen_symbol(piece_str)
            rank_pieces.append(symbol)
        fen_rank_str = ""
        empty_count = 0
        for s in rank_pieces:
            if s == "":
                empty_count += 1
            else:
                if empty_count > 0:
                    fen_rank_str += str(empty_count)
                    empty_count = 0
                fen_rank_str += s
        if empty_count > 0:
            fen_rank_str += str(empty_count)
        fen_ranks.append(fen_rank_str)
    fen_string = "/".join(fen_ranks)
    return fen_string

def download_image(url):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        image_data = np.frombuffer(response.content, np.uint8)
        img = cv2.imdecode(image_data, cv2.IMREAD_UNCHANGED) # Keep alpha channel
        return img
    else:
        raise Exception(f"Failed to download image from {url}")

def initialize_piece_images():
    """Initialize all piece images at startup"""
    global PIECE_IMAGES
    
    # Mapping from fen char to piece code used in the URL
    fen_to_code = {
        'K': 'wk', 'Q': 'wq', 'R': 'wr', 'B': 'wb', 'N': 'wn', 'P': 'wp',
        'k': 'bk', 'q': 'bq', 'r': 'br', 'b': 'bb', 'n': 'bn', 'p': 'bp'
    }

    base_url = "https://www.chess.com/chess-themes/pieces/neo/300/"
    
    print("Loading piece images...")
    for fen_char, code in fen_to_code.items():
        url = base_url + code + ".png"
        try:
            PIECE_IMAGES[fen_char] = download_image(url)
            print(f"Loaded {code}")
        except Exception as e:
            print(f"Failed to load {code}: {str(e)}")
    
    print("Finished loading piece images")
    return PIECE_IMAGES

def overlay_image_alpha(background, overlay, x, y):
    bh, bw = background.shape[:2]
    oh, ow = overlay.shape[:2]

    if x + ow > bw or y + oh > bh or x < 0 or y < 0:
        return

    alpha_channel = overlay[:,:,3] / 255.0
    for c in range(3):
        background[y:y+oh, x:x+ow, c] = (alpha_channel * overlay[:,:,c] +
                                        (1 - alpha_channel) * background[y:y+oh, x:x+ow, c])

def draw_chessboard(fen, square_size=80):
    """Modified to use global PIECE_IMAGES"""
    board_size = 8
    board_img = np.zeros((board_size * square_size, board_size * square_size, 3), dtype=np.uint8)

    light_color = (240, 217, 181)
    dark_color = (181, 136, 99)

    ranks = fen.split('/')
    for r in range(board_size):
        for c in range(board_size):
            color = dark_color if (r + c) % 2 else light_color
            y1, x1 = r * square_size, c * square_size
            cv2.rectangle(board_img, (x1, y1), (x1 + square_size, y1 + square_size), color, -1)

    # Place pieces
    for rank_idx, rank_str in enumerate(ranks):
        col_idx = 0
        for char in rank_str:
            if char.isdigit():
                col_idx += int(char)
            else:
                piece = char
                piece_img = PIECE_IMAGES.get(piece, None)
                if piece_img is not None:
                    # Scale piece to about ~80% of the square size
                    desired_size = int(square_size * 0.8)
                    scaled = cv2.resize(piece_img, (desired_size, desired_size), interpolation=cv2.INTER_AREA)

                    # Calculate position to center the piece
                    y_center = rank_idx * square_size + (square_size - desired_size) // 2
                    x_center = col_idx * square_size + (square_size - desired_size) // 2

                    overlay_image_alpha(board_img, scaled, x_center, y_center)

                col_idx += 1
    return board_img

def get_fen():
    fen_examples = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",  # Starting position
        "rnbq1bnr/pppppppp/8/4k3/8/4K3/PPPPPPPP/RNBQ1BNR w KQ - 0 1",  # Kings in the center
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 1 1",  # Black to move
        "r3k2r/pppqppbp/2n1bnp1/3p4/3P4/2N1BNP1/PPPQPPBP/R3K2R w KQkq - 0 10",  # Mid-game position
        "rnb1kbnr/pppp1ppp/4p3/8/8/4P3/PPPP1PPP/RNB1KBNR w KQkq - 0 1",  # Random valid position
        "rnbqkb1r/pppppppp/5n2/8/8/5N2/PPPPPPPP/RNBQKB1R w KQkq - 0 1",  # Knights moved
        "r5rk/5p1p/5R2/4B3/8/8/7P/7K w - - 0 1", # white mate in 3
        "rnb1kbnr/pppp1ppp/8/4p3/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 2" #white dominating
    ]
    return random.choice(fen_examples)

def draw_eval_bar(board_img, eval_score, bar_width=30):
    board_h, board_w, _ = board_img.shape
    base_color = (169, 169, 169) 
    eval_color = (255, 255, 255) 

    bar_img = np.ones((board_h, bar_width, 3), dtype=np.uint8)
    bar_img[:] = base_color

    eval_normalized = max(min(eval_score, 10), -10) 
    eval_height = int(((eval_normalized + 10) / 20) * board_h)

    bar_img[board_h - eval_height:, :] = eval_color

    return np.hstack((board_img, bar_img))


def draw_arrow(board_img, from_square, to_square, square_size=80, color=(0, 165, 255), thickness=3):
    from_row, from_col = 8 - int(from_square[1]), ord(from_square[0]) - ord('a')
    to_row, to_col = 8 - int(to_square[1]), ord(to_square[0]) - ord('a')


    from_x = int(from_col * square_size + square_size / 2)
    from_y = int(from_row * square_size + square_size / 2)
    to_x = int(to_col * square_size + square_size / 2)
    to_y = int(to_row * square_size + square_size / 2)

    # Draw arrow
    cv2.arrowedLine(board_img, (from_x, from_y), (to_x, to_y), color, thickness, tipLength=0.4)

    return board_img

def main():
    # Initialize piece images once at startup
    initialize_piece_images()
    
    # Make sure temp_squares directory exists
    os.makedirs("temp_squares", exist_ok=True)
    
    # Example: Adjust src_points to match your board setup
    src_points = np.array([
        [952, 438],
        [2978, 430],
        [3455, 2524],
        [625, 2605]
    ], dtype="float32")

    # Gather all images
    image_directory = "game_images"
    image_files = sorted([f for f in os.listdir(image_directory) if f.endswith('.jpg')])

    fen_states = []
    evals = []
    next_moves = []
    for i, filename in enumerate(image_files, start=1):
        print(f"Processing image {i}/{len(image_files)}: {filename}")
        image_path = os.path.join(image_directory, filename)
        original_image = cv2.imread(image_path)
        if original_image is None:
            raise FileNotFoundError(f"Could not load image at {image_path}")
            
        warped_image, expanded_warped = load_and_warp(image_path, src_points, expansion_factor=1.2)
        squares = divide_into_squares(warped_image, expanded_warped, src_points)

        squares_images_paths = []
        for row_idx, row in enumerate(squares):
            row_paths = []
            for col_idx, square in enumerate(row):
                square_path = f"temp_squares/square_{i}_{row_idx}_{col_idx}.jpg"
                cv2.imwrite(square_path, square)
                row_paths.append(square_path)
            squares_images_paths.append(row_paths)

        # fen = generate_fen_from_predictions(squares_images_paths)
        fen = get_fen()
        print(f"Generated FEN: {fen}")
        fen_states.append(fen)

        chess_api_url = "https://chess-api.com/v1"
        depth = 15

        payload = {
            "fen": fen,
            "depth": depth
        }
        try:
            response = requests.post(chess_api_url, json=payload)
            response.raise_for_status()
            data = response.json()
            if data.get('type') == 'error':
                raise Exception(data.get('text'))
            eval = data.get('eval')
            move_set = data.get('continuationArr')
            print(f"Next move: {move_set[0]}")
            # print(f"Next: {data.get('continuationArr')}")

            next_move = data.get('continuationArr', [])[0] if data.get('continuationArr') else None
            # print(f"Eval: {eval}")
            evals.append(eval)
            next_moves.append(next_move)
            # print(data)
        except requests.exceptions.RequestException as e:
            print(f"Error: {str(e)}")

    # Interactive Viewer
    index = 0
    show_next = False
    while True:
        fen = fen_states[index]
        eval = evals[index]
        board_img = draw_chessboard(fen, square_size=80)
        board_img_with_eval = draw_eval_bar(board_img, eval)

        if show_next:
            next_move = next_moves[index]
            from_square = next_move[:2]
            to_square = next_move[2:]
            board_img_with_eval = draw_arrow(board_img_with_eval, from_square, to_square)

        cv2.putText(board_img_with_eval, f"Eval: {eval}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(board_img, fen, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.imshow("Virtual Chessboard Viewer", board_img_with_eval)
        

        # cv2.imshow("Virtual Chessboard Viewer", board_img)
        key = cv2.waitKey(0)
        if key == ord('q'): # Quit
            break
        elif key == ord('n') or key == 83: # 'n' or right arrow
            index = min(index + 1, len(fen_states)-1)
        elif key == ord('p') or key == 81: # 'p' or left arrow
            index = max(index - 1, 0)
        elif key == ord('m'):
            show_next = not show_next

    cv2.destroyAllWindows()
    
    # Cleanup temp files
    for file in os.listdir("temp_squares"):
        os.remove(os.path.join("temp_squares", file))

if __name__ == "__main__":
    main()