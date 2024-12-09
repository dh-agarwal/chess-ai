import ollama

def detect_piece(img_path):
    prompt = "This is a zoomed in image of a square on a chess board. If the square is empty, respond with the word 'None'. If there is a piece on the square, respond with the color of the piece, followed by a space, followed by the name of the piece. For example, your responses may look like:\n'white rook'\n'black bishop'\n'white king\nEnsure that your responses do not have any additional content - do not include full sentences, periods, or otherwise extraneous information. Additionally, your responses should be in all lowercase. Your responses should NOT look like:\n'This square is empty'\n'Black pawn'.\n'There is nothing on this square'"
    response = ollama.chat(
        model='llama3.2-vision:11b',
        messages=[{
            'role': 'user',
            'content': prompt,
            'images': [img_path]
        }]
    )

    return response

response = detect_piece('./image.png')
print(response.message.content)