import ollama
import time
def detect_piece(img_path):
    llama3_11b = "This is a zoomed in image of a single square on a chess board. If the square is empty, respond with the word 'None'. If there is a piece on the square, respond with the color of the piece, followed by the name of the piece. For example, your responses may look like:\n'white rook'\n'black bishop'\n'white king\nEnsure that your responses do not have any additional content - do not include full sentences, periods, or otherwise extraneous information. Additionally, your responses should be in all lowercase. Your responses should NOT look like:\n'This square is empty'\n'Black pawn'.\n'There is nothing on this square'"
    moondream_1p8b = "This is a picture of a single square on a chess board. Respond with the color and name of the piece (i.e. 'white knight'), or 'None' if there is no piece."
    response = ollama.chat(
        model='llava-llama3',
        messages=[{
            'role': 'user',
            'content': llama3_11b,
            'images': [img_path]
        }]
    )

    return response


start = time.time()
for i in range(64):
    response = detect_piece('./image.png')
end = time.time()
print("time:", end - start)
print("avg per req:", (end - start)/64.0)
print(response)