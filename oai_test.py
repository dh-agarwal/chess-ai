from openai import OpenAI
import base64
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Function to get the response from OpenAI based on the image
def get_chess_piece_response(image_path):
    base64_image = encode_image(image_path)
    prompt = "If the square is empty, respond with the word 'None'. If there is a piece on the square, respond with the color of the piece, followed by the name of the piece. For example, your responses may look like:\n'white rook'\n'black bishop'\n'white king\nEnsure that your responses do not have any additional content - do not include full sentences, periods, or otherwise extraneous information. Additionally, your responses should be in all lowercase. Your responses should NOT look like:\n'This square is empty'\n'Black pawn'.\n'There is nothing on this square'"

    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=0,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "This is a zoomed in image of a single square on a chess board." },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                        },
                    },
                    {"type": "text", "text": prompt}

                ],
            }
        ],
        max_tokens=300,
    )

    input_tokens = response.usage.total_tokens  # Assuming the response contains usage info
    output_tokens = response.usage.completion_tokens  # Assuming the response contains usage info
    print(f"Input tokens: {input_tokens}, Output tokens: {output_tokens}")

    return response.choices[0]
import time
import concurrent.futures

def run_parallel_requests(image_path, num_requests=64):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(get_chess_piece_response, image_path) for _ in range(num_requests)]
        responses = [future.result() for future in concurrent.futures.as_completed(futures)]
    return responses

start = time.time()
responses = run_parallel_requests("image copy 2.png")
end = time.time()

for resp in responses:
    print(resp)
print("time:", end - start)