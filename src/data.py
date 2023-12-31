import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import open_clip
from diffusers import AudioLDMPipeline
import json
import csv

CLIP = 0
CLAP = 0
TOKENIZER = 0

def get_models():
    global CLIP
    global CLAP
    global TOKENIZER

    if not CLIP:
        CLIP, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
        CLIP.to("cuda")
    
    if not TOKENIZER:
        TOKENIZER = open_clip.get_tokenizer('ViT-B-32')

    if not CLAP:
        repo_id = "cvssp/audioldm-l-full"
        CLAP = AudioLDMPipeline.from_pretrained(repo_id, torch_dtype=torch.float16)
        CLAP.to("cuda")
    
    return CLIP, CLAP, TOKENIZER

def get_data_gcc():
    # Google Conceptual Captions
    train_data = []
    with open("../data/Train-GCC.tsv") as file:
        tsv_file = csv.reader(file, delimiter="\t")
        # printing data line by line
        for line in tsv_file:
            train_data.append(line)
        train_data = [d[0] for d in train_data]

    test_data = []
    with open("../data/Validation-GCC.tsv") as file:
        tsv_file = csv.reader(file, delimiter="\t")
        # printing data line by line
        for line in tsv_file:
            test_data.append(line)
        test_data = [d[0] for d in test_data]

    print(f"Train data head: {train_data[:5]}")
    print(f"Test data head: {test_data[:5]}")

    return train_data, test_data

def get_data_coco():
    train_file = open('../data/captions_train2017.json')
    train_data = json.load(train_file)
    train_data = [d["caption"] for d in train_data["annotations"]]

    test_file = open('../data/captions_val2017.json')
    test_data = json.load(test_file)
    test_data = [d["caption"] for d in test_data["annotations"]]
    return train_data, test_data

def get_embeds(data):
  clip_tokens = TOKENIZER(data).to("cuda")

  with torch.no_grad(), torch.cuda.amp.autocast():
    clip_embeds = CLIP.encode_text(clip_tokens).to("cuda")
    nn.functional.normalize(clip_embeds, p=2, dim=1)

    clap_embeds = CLAP._encode_prompt(
        prompt = data,
        num_waveforms_per_prompt = 1,
        do_classifier_free_guidance = False,
        device="cuda"
    ).to("cuda")

  return clip_embeds.float(), clap_embeds.float()

class EmbeddingsDataset(Dataset):
    def __init__(self, data, transform=None, target_transform=None):
        self.data = data
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        with torch.no_grad():
            clip_embed, clap_embed = get_embeds(self.data[idx])
            if self.transform:
                clip_embed = self.transform(clip_embed)
            if self.target_transform:
                clap_embed = self.target_transform(clap_embed)

        return clip_embed.squeeze(0), clap_embed.squeeze(0)
