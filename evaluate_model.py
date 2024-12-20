import torch
from train_custom import ChessPieceClassifier, transform
from PIL import Image
import json
import matplotlib.pyplot as plt
from collections import defaultdict

def load_model(model_path, num_classes=13):
    model = ChessPieceClassifier(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def evaluate_model(model, json_path):
    # Initialize counters
    class_totals = defaultdict(int)
    class_correct = defaultdict(int)
    
    # Create reverse label mapping
    label_counts = defaultdict(int)
    with open(json_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            if data["use"] == 1:  # Only consider images marked for use
                label_counts[data["label"]] += 1
    
    label_to_idx = {label: idx for idx, label in enumerate(sorted(label_counts.keys()))}
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}
    
    # Evaluate each image
    with open(json_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            if data["use"] != 1:
                continue
                
            try:
                # Load and transform image
                image = Image.open(data["img_path"]).convert('RGB')
                img_tensor = transform(image).unsqueeze(0)
                
                # Get prediction
                with torch.no_grad():
                    outputs = model(img_tensor)
                    _, predicted = outputs.max(1)
                
                true_label = label_to_idx[data["label"]]
                pred_label = predicted.item()
                
                # Update counters
                class_totals[data["label"]] += 1
                if true_label == pred_label:
                    class_correct[data["label"]] += 1
                    
            except Exception as e:
                print(f"Error processing {data['img_path']}: {str(e)}")
                continue
    
    return class_correct, class_totals

def plot_results(class_correct, class_totals):
    labels = sorted(class_totals.keys())
    accuracies = [class_correct[label] / class_totals[label] * 100 for label in labels]
    
    # Create bar plot
    plt.figure(figsize=(12, 6))
    bars = plt.bar(labels, accuracies)
    plt.title('Classification Accuracy by Chess Piece')
    plt.xlabel('Piece Type')
    plt.ylabel('Accuracy (%)')
    plt.xticks(rotation=45)
    
    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    # Print detailed results
    print("\nDetailed Results:")
    print("----------------")
    total_correct = sum(class_correct.values())
    total_samples = sum(class_totals.values())
    
    for label in sorted(class_totals.keys()):
        correct = class_correct[label]
        total = class_totals[label]
        accuracy = (correct / total) * 100
        print(f"{label}: {correct}/{total} ({accuracy:.1f}%)")
    
    print("\nOverall Accuracy: {}/{} ({:.1f}%)".format(
        total_correct, total_samples, (total_correct/total_samples)*100))

if __name__ == "__main__":
    model = load_model("chess_piece_classifier_epoch_40.pth")
    class_correct, class_totals = evaluate_model(model, "labels.jsonl")
    plot_results(class_correct, class_totals)