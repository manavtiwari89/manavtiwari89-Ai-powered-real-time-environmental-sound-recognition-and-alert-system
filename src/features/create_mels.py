import os
import torchaudio

INPUT_DIR = "data/processed/chunks"
OUTPUT_DIR = "data/processed/mels"

SAMPLE_RATE = 16000
N_FFT = 1024
HOP_LENGTH = 256
N_MELS = 64

mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    n_mels=N_MELS
)

db_transform = torchaudio.transforms.AmplitudeToDB()

for split in ["train", "val", "test"]:

    split_dir = os.path.join(INPUT_DIR, split)

    for category in os.listdir(split_dir):

        category_dir = os.path.join(split_dir, category)

        if not os.path.isdir(category_dir):
            continue

        output_category_dir = os.path.join(
            OUTPUT_DIR, split, category
        )

        os.makedirs(output_category_dir, exist_ok=True)

        for filename in os.listdir(category_dir):

            if not filename.endswith(".wav"):
                continue

            input_path = os.path.join(category_dir, filename)

            waveform, sample_rate = torchaudio.load(input_path)

            # Convert to mono
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            # Resample if necessary
            if sample_rate != SAMPLE_RATE:
                resampler = torchaudio.transforms.Resample(
                    sample_rate,
                    SAMPLE_RATE
                )
                waveform = resampler(waveform)

            # Create Mel Spectrogram
            mel = mel_transform(waveform)

            # Convert to decibels
            mel_db = db_transform(mel)

            # Save as PyTorch tensor
            output_filename = (
                os.path.splitext(filename)[0] + ".pt"
            )

            output_path = os.path.join(
                output_category_dir,
                output_filename
            )

            torch_save = mel_db.squeeze(0)

            import torch
            torch.save(torch_save, output_path)

print("Mel Spectrogram creation completed.")
