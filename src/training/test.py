import os
import sys
import torch
from torch.utils.data import DataLoader

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

from training.dataset import SoundDataset, CLASS_NAMES
from models.cnn import SoundCNN


# Device
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("Using device:", device)


# Test dataset
test_dataset = SoundDataset(
    "data/processed/mels/test",
    CLASS_NAMES
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)


# Model
model = SoundCNN(
    num_classes=len(CLASS_NAMES)
).to(device)

model.load_state_dict(
    torch.load(
        "models/best_cnn.pt",
        map_location=device,
        weights_only=True
    )
)

model.eval()


# Evaluation
correct = 0
total = 0

with torch.no_grad():

    for inputs, labels in test_loader:

        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)

        predictions = outputs.argmax(dim=1)

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)


accuracy = 100 * correct / total

print()
print("Test samples:", total)
print(f"Test accuracy: {accuracy:.2f}%")
