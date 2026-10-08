import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader


class SoundDataset(Dataset):

    def __init__(self, root_dir, class_names):

        self.files = []
        self.labels = []

        self.class_names = class_names
        self.class_to_index = {
            name: index
            for index, name in enumerate(class_names)
        }

        for class_name in class_names:

            class_dir = os.path.join(root_dir, class_name)

            files = glob.glob(
                os.path.join(class_dir, "*.pt")
            )

            for file in files:
                self.files.append(file)
                self.labels.append(
                    self.class_to_index[class_name]
                )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):

        spectrogram = torch.load(
            self.files[index],
            weights_only=True
        )

        # Add channel dimension
        # [64, 63] → [1, 64, 63]
        spectrogram = spectrogram.unsqueeze(0)

        label = self.labels[index]

        return spectrogram, label


CLASS_NAMES = [
    "clapping",
    "clock_alarm",
    "crying_baby",
    "dog",
    "door_wood_knock",
    "glass_breaking",
    "laughing",
    "siren",
    "doorbell"
]


if __name__ == "__main__":

    train_dataset = SoundDataset(
        "data/processed/mels/train",
        CLASS_NAMES
    )

    val_dataset = SoundDataset(
        "data/processed/mels/val",
        CLASS_NAMES
    )

    test_dataset = SoundDataset(
        "data/processed/mels/test",
        CLASS_NAMES
    )

    print("Train samples:", len(train_dataset))
    print("Validation samples:", len(val_dataset))
    print("Test samples:", len(test_dataset))

    x, y = train_dataset[0]

    print("Sample shape:", x.shape)
    print("Sample label:", y)

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True
    )

    batch_x, batch_y = next(iter(train_loader))

    print("Batch shape:", batch_x.shape)
    print("Labels shape:", batch_y.shape)
