import torch
from transformers import OwlViTProcessor, OwlViTForObjectDetection
from PIL import Image
import time
processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")

model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")

image = Image.open("image.png").convert("RGB")

texts = [["a photo blank square", "a photo of a black pawn", "a photo of a white pawn", "a photo of a black rook", "a photo of a white rook", "a photo of a black knight", "a photo of a white knight", "a photo of a black bishop", "a photo of a white bishop", "a photo of a black queen", "a photo of a white queen", "a photo of a black king", "a photo of a white king"]]

inputs = processor(text=texts, images=image, return_tensors="pt")
start = time.time()
outputs = model(**inputs)
end = time.time()
print("time:", end - start)
# Target image sizes (height, width) to rescale box predictions [batch_size, 2]

target_sizes = torch.Tensor([image.size[::-1]])

# Convert outputs (bounding boxes and class logits) to Pascal VOC format (xmin, ymin, xmax, ymax)

results = processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=0.1)

i = 0  # Retrieve predictions for the first image for the corresponding text queries

text = texts[i]

boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]

for box, score, label in zip(boxes, scores, labels):
    box = [round(i, 2) for i in box.tolist()]
    print(f"Detected {text[label]} with confidence {round(score.item(), 3)} at location {box}")