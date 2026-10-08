import os
import sys
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report

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


# Load best model
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


# Predictions
all_labels = []
all_predictions = []

with torch.no_grad():

    for inputs, labels in test_loader:

        inputs = inputs.to(device)

        outputs = model(inputs)

        predictions = outputs.argmax(dim=1)

        all_labels.extend(labels.numpy())
        all_predictions.extend(
            predictions.cpu().numpy()
        )


# Confusion matrix
cm = confusion_matrix(
    all_labels,
    all_predictions
)

print("\nConfusion Matrix:")
print(cm)


# Classification report
print("\nClassification Report:\n")

print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=CLASS_NAMES,
        digits=2,
        zero_division=0
    )
)
