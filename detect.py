import cv2
import numpy as np
import os
import requests
import torch
from PIL import Image
from torchvision import transforms
import torch.nn as nn
import torchvision.models as models

PIECE_IMAGES = {}

class ChessPieceClassifier(nn.Module):
    def __init__(self, num_classes=13):
        super(ChessPieceClassifier, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.resnet(x)
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ChessPieceClassifier(num_classes=13).to(device)
model.load_state_dict(torch.load("chess_piece_classifier_epoch_40.pth", map_location=device))
model.eval()

def load_and_warp(image_path, src_points, dst_size=750, expansion_factor=1.08):
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

    center = np.mean(src_points, axis=0)
    expanded_src_points = np.array([
        src_points[0] + np.array([(src_points[0][0] - center[0]) * (expansion_factor - 1), 
                                (src_points[0][1] - center[1]) * (expansion_factor - 1)]),
        src_points[1] + np.array([(src_points[1][0] - center[0]) * (expansion_factor - 1), 
                                (src_points[1][1] - center[1]) * (expansion_factor - 1)]),
        src_points[2] + np.array([(src_points[2][0] - center[0]) * (expansion_factor - 1),
                                (src_points[2][1] - center[1]) * (expansion_factor - 1)]),
        src_points[3] + np.array([(src_points[3][0] - center[0]) * (expansion_factor - 1),
                                (src_points[3][1] - center[1]) * (expansion_factor - 1)])
    ], dtype="float32")

    expanded_transform = cv2.getPerspectiveTransform(expanded_src_points, dst_points)
    expanded_warped = cv2.warpPerspective(image, expanded_transform, (dst_size, dst_size))

    return warped, expanded_warped

def divide_into_squares(warped_image, expanded_warped, board_size=8):
    h, w = warped_image.shape[:2]
    square_h, square_w = h // board_size, w // board_size

    regular_vertical_padding = int(square_h * 0.25)
    downward_padding = int(square_h * 0.15)
    regular_horizontal_padding = int(square_w * 0.3)
    border_vertical_padding = int(square_h * 0.25)
    border_horizontal_padding = int(square_w * 0.25)

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

            padded_x1 = max(0, x1 - horiz_padding)
            padded_x2 = min(w, x2 + horiz_padding)
            padded_y1 = max(0, y1 - vert_padding)
            padded_y2 = min(h, y2 + downward_padding)

            square = source_img[padded_y1:padded_y2, padded_x1:padded_x2]
            row.append(square)

        squares.append(row)
    return squares

def detect_piece(img_path):
    inv_label_mapping = {
        0: "black bishop",
        1: "black king",
        2: "black knight",
        3: "black pawn",
        4: "black queen",
        5: "black rook",
        6: "none",
        7: "white bishop",
        8: "white king",
        9: "white knight",
        10: "white pawn",
        11: "white queen",
        12: "white rook"
    }

    transform_inference = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.51054407, 0.4892153, 0.46805718],
                            std=[0.51054407, 0.4892153, 0.46805718])
    ])
    
    image = Image.open(img_path).convert('RGB')
    image = transform_inference(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
        _, predicted = outputs.max(1)
    piece_label = inv_label_mapping[predicted.item()]
    return piece_label

def piece_to_fen_symbol(piece_str):
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
        img = cv2.imdecode(image_data, cv2.IMREAD_UNCHANGED)
        return img
    else:
        raise Exception(f"Failed to download image from {url}")

def initialize_piece_images():
    global PIECE_IMAGES
    fen_to_code = {
        'K': 'wk', 'Q': 'wq', 'R': 'wr', 'B': 'wb', 'N': 'wn', 'P': 'wp',
        'k': 'bk', 'q': 'bq', 'r': 'br', 'b': 'bb', 'n': 'bn', 'p': 'bp'
    }
    base_url = "https://www.chess.com/chess-themes/pieces/neo/300/"
    for fen_char, code in fen_to_code.items():
        url = base_url + code + ".png"
        try:
            PIECE_IMAGES[fen_char] = download_image(url)
        except Exception as e:
            print(f"Failed to load {code}: {str(e)}")
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
            
    for rank_idx, rank_str in enumerate(ranks):
        col_idx = 0
        for char in rank_str:
            if char.isdigit():
                col_idx += int(char)
            else:
                piece = char
                piece_img = PIECE_IMAGES.get(piece, None)
                if piece_img is not None:
                    desired_size = int(square_size * 0.8)
                    scaled = cv2.resize(piece_img, (desired_size, desired_size), interpolation=cv2.INTER_AREA)
                    y_center = rank_idx * square_size + (square_size - desired_size) // 2
                    x_center = col_idx * square_size + (square_size - desired_size) // 2
                    overlay_image_alpha(board_img, scaled, x_center, y_center)
                col_idx += 1
                
    return board_img

def extract_number(fname):
    return int(os.path.splitext(fname)[0])

def main():
    initialize_piece_images()
    os.makedirs("temp_squares", exist_ok=True)
    
    src_points = np.array([
        [446, 45],
        [1405, 44],
        [1407, 1015],
        [429, 1009]
    ], dtype="float32")
    
    image_directory = "game_images/game_2"
    image_files = [f for f in os.listdir(image_directory) if f.endswith('.png')]
    image_files.sort(key=extract_number)

    if len(image_files) < 2:
        print("Need at least 2 images to detect white's side")
        return
    fen_states = []
    
    for i, filename in enumerate(image_files, start=1):
        image_path = os.path.join(image_directory, filename)
        warped_image, expanded_warped = load_and_warp(image_path, src_points)
        squares = divide_into_squares(warped_image, expanded_warped)
        squares_images_paths = []
        for row_idx, row in enumerate(squares):
            row_paths = []
            for col_idx, square in enumerate(row):
                square_path = f"temp_squares/square_{i}_{row_idx}_{col_idx}.jpg"
                cv2.imwrite(square_path, square)
                row_paths.append(square_path)
            squares_images_paths.append(row_paths)
        fen = generate_fen_from_predictions(squares_images_paths)
        fen_states.append(fen)
        
    index = 0
    cv2.namedWindow("Virtual Chessboard Viewer", cv2.WINDOW_AUTOSIZE)

    while True:
        fen = fen_states[index]
        board_img = draw_chessboard(fen, square_size=80)
        cv2.setWindowTitle("Virtual Chessboard Viewer", f"Virtual Chessboard Viewer - Move: {index + 1}")
        cv2.imshow("Virtual Chessboard Viewer", board_img)
        key = cv2.waitKey(0)
        if key == ord('q'):
            break
        elif key == ord('n') or key == 83:
            index = min(index + 1, len(fen_states)-1)
        elif key == ord('p') or key == 81:
            index = max(index - 1, 0)
            
    cv2.destroyAllWindows()
    
    for file in os.listdir("temp_squares"):
        os.remove(os.path.join("temp_squares", file))

if __name__ == "__main__":
    main()
