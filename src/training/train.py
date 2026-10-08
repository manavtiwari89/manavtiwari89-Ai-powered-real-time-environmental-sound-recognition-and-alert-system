import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

from training.dataset import SoundDataset, CLASS_NAMES
from models.cnn import SoundCNN


# --------------------------------------------------
# Device
# --------------------------------------------------

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("Using device:", device)


# --------------------------------------------------
# Settings
# --------------------------------------------------

BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 20


# --------------------------------------------------
# Datasets
# --------------------------------------------------

train_dataset = SoundDataset(
    "data/processed/mels/train",
    CLASS_NAMES
)

val_dataset = SoundDataset(
    "data/processed/mels/val",
    CLASS_NAMES
)


# --------------------------------------------------
# DataLoaders
# --------------------------------------------------

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# --------------------------------------------------
# Model
# --------------------------------------------------

model = SoundCNN(
    num_classes=len(CLASS_NAMES)
).to(device)


# --------------------------------------------------
# Loss + Optimizer
# --------------------------------------------------

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# --------------------------------------------------
# Training
# --------------------------------------------------

best_val_accuracy = 0.0

os.makedirs("models", exist_ok=True)

for epoch in range(EPOCHS):

    # -------------------------
    # Training phase
    # -------------------------

    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:

        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        predictions = outputs.argmax(dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    train_loss = total_loss / len(train_loader)
    train_accuracy = 100 * correct / total


    # -------------------------
    # Validation phase
    # -------------------------

    model.eval()

    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for inputs, labels in val_loader:

            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)

            loss = criterion(outputs, labels)

            val_loss += loss.item()

            predictions = outputs.argmax(dim=1)

            val_correct += (
                (predictions == labels).sum().item()
            )

            val_total += labels.size(0)

    val_loss /= len(val_loader)
    val_accuracy = 100 * val_correct / val_total


    # -------------------------
    # Print results
    # -------------------------

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Train Loss: {train_loss:.4f} "
        f"Train Acc: {train_accuracy:.2f}% "
        f"Val Loss: {val_loss:.4f} "
        f"Val Acc: {val_accuracy:.2f}%"
    )


    # -------------------------
    # Save best model
    # -------------------------

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        torch.save(
            model.state_dict(),
            "models/best_cnn.pt"
        )

        print(
            f"  → Best model saved "
            f"(Val Acc: {val_accuracy:.2f}%)"
        )


print()
print("Training completed.")
print(
    f"Best validation accuracy: "
    f"{best_val_accuracy:.2f}%"
)
