import json
from pathlib import Path

import torch
import torchaudio


class SpeechData(torch.utils.data.Dataset):
    def __init__(self, manifest_dir, root_dir=None):
        if root_dir is None:
            root_dir = "."
        self.root_dir = Path(root_dir)
        self.manifest_dir = self.root_dir / manifest_dir
        self.wav_field = "wav_path"
        self.manifest = [json.loads(line.strip()) for line in open(manifest_dir)]

    def __getitem__(self, idx):
        wav_path = self.root_dir / self.manifest[idx][self.wav_field]
        return {
            "wav": torchaudio.load(wav_path)[0].squeeze(0),
            "id": str(self.manifest[idx]["id"]),
        }

    def __len__(self):
        return len(self.manifest)


class TextData(torch.utils.data.Dataset):
    def __init__(self, manifest_dir, root_dir=None):
        if root_dir is None:
            root_dir = "."
        self.root_dir = Path(root_dir)
        self.manifest_dir = self.root_dir / manifest_dir
        self.text_field = "asr"
        self.manifest = [json.loads(line.strip()) for line in open(manifest_dir)]

    def __getitem__(self, idx):
        return {
            "text": self.manifest[idx][self.text_field],
            "id": str(self.manifest[idx]["id"]),
        }

    def __len__(self):
        return len(self.manifest)
