import os
import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig
import numpy as np

from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image, ImageOps
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import random
from torch.optim import AdamW
import torch.nn.functional as F
from scipy.stats import kendalltau
import re
import ast
import timm
from torchvision.transforms import InterpolationMode
import lpips

loss_fn = lpips.LPIPS(net='alex')  # Options: 'alex', 'vgg', 'squeeze'



# from ViT_siamese import ViTSiameseNetwork
transform = transforms.Compose([
    transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def pad_images(images, max_length):
    """
    Pads the images to ensure that each sample has the same number of images.
    """
    current_length = len(images)
    if current_length < max_length:
        padding = [torch.zeros_like(images[0])] * (max_length - current_length)
        images = images + padding  # Pad with zeros (or any other padding value)
    return torch.stack(images)  # Stack into a tensor

def generate_splits(all_samples, ratio = 0.85):
    # random.shuffle(all_samples)

    # compute split point at 85%
    split_idx = int(ratio * len(all_samples))

    train_samples = all_samples[:split_idx]
    val_samples   = all_samples[split_idx:]
    return train_samples, val_samples


def pad_tensor_list(tensors, max_len, pad_value=-1):
    """
    Pads a list of tensors to the same length.
    Assumes tensors are of shape [C, H, W] or [D].
    """
    padded = tensors + [torch.zeros_like(tensors[0]).fill_(pad_value)] * (max_len - len(tensors))
    return torch.stack(padded)

def resize_and_pad(image, target_size=224, padding_color=(0, 0, 0)):
    """
    Resize an image so the long side is `target_size`, then pad the short side to `target_size`.
    
    Args:
        image (PIL.Image): Input image.
        target_size (int): Target size for the long side.
        padding_color (tuple): RGB color for padding.
    
    Returns:
        PIL.Image: Resized and padded image of shape (target_size, target_size).
    """
    # Resize so the long side becomes target_size
    w, h = image.size
    if w > h:
        new_w = target_size
        new_h = int(target_size * h / w)
    else:
        new_h = target_size
        new_w = int(target_size * w / h)
    image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    # Calculate padding
    delta_w = target_size - new_w
    delta_h = target_size - new_h
    padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
    
    # Pad the image
    padded_image = ImageOps.expand(image, padding, fill=padding_color)
    return padded_image



class SiameseDataset(Dataset):
    def __init__(self, image_label_sets, transform=None):
        """
        image_label_pairs: List of (image_path, label) pairs
        transform: Image transform (e.g., resize, normalize)
        """
        self.paths = image_label_sets
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        try:
            # print(self.image_label_pairs[idx])
            original_path = self.paths[idx][0]
            manipulations_path = []
            labels = []
            for data_point in self.paths[idx][1:]:
                manipulations_path.append(data_point[0])
                labels.append(int(data_point[1]))


            original = Image.open(original_path).convert("RGB")
            original = resize_and_pad(original)

            manipulations = []
            for manipulation in manipulations_path:
                img = Image.open(manipulation).convert("RGB")
                img = resize_and_pad(img)
                manipulations.append(img)

            if self.transform:
                manipulations_torch = []
                original = self.transform(original)
                for index in range(len(manipulations)):
                    manipulations_torch.append(self.transform(manipulations[index]))

            prev_amount = len(manipulations_torch)
            max_images = 2  # Set the max number of images in img2 for all samples
            manipulations_torch = pad_images(manipulations_torch, max_images)  # Pad img2 to the same length
            for i in range(2 - (len(labels))):
                labels.append(-1)
            return original, manipulations_torch, torch.tensor(labels, dtype=torch.float32), prev_amount, original_path, manipulations_path
        except:
            print(f"failed on image: {self.authentic_images[idx]} with images {self.manipulated_images[idx]}")

###MODEL
class ViTSiameseNetwork(nn.Module):
    def __init__(self, pretrained_model='google/vit-base-patch16-224', dropping = False):
        super(ViTSiameseNetwork, self).__init__()
        if dropping == True:
            config = ViTConfig.from_pretrained(
                "google/vit-base-patch16-224",
                drop_path_rate=0.1  # stochastic depth rate
            )

            # Load model with modified config
            self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224", config=config)
        else:
            self.vit = ViTModel.from_pretrained(pretrained_model)
        self.embedding_dim = self.vit.config.hidden_size
        self.log_sigma1 = nn.Parameter(torch.tensor(0.0))
        self.log_sigma2 = nn.Parameter(torch.tensor(0.0))

    def forward_once(self, x):
        outputs = self.vit(pixel_values=x)
        tokens = outputs.last_hidden_state   # (B, N, D)
        return tokens

    def forward(self, x1, x2):
        feat1 = self.forward_once(x1)
        b,c = x2.shape[0], x2.shape[1]
        x2_flat = x2.view(b*c, x2.shape[2], x2.shape[3], x2.shape[4])
        feat2 = self.forward_once(x2_flat)
        out = feat2.view(b,c, 197, -1)

        return feat1, out


class DINOViTSiameseNetwork(nn.Module):
    def __init__(self, model_name='vit_base_patch16_224_dino'):
        super(DINOViTSiameseNetwork, self).__init__()

        # Load DINO ViT from timm
        self.vit = timm.create_model(model_name, pretrained=True)
        self.vit.reset_classifier(0)  # Remove classification head

        self.embedding_dim = self.vit.embed_dim  # Usually 768

    def forward_once(self, x):
        tokens = self.vit.forward_features(x)  # Shape: [B, N, D]
        return tokens

    def forward(self, x1, x2):
        feat1 = self.forward_once(x1)  # [B, N, D]

        b, c = x2.shape[0], x2.shape[1]
        x2_flat = x2.view(b * c, x2.shape[2], x2.shape[3], x2.shape[4])  # [B*C, 3, 224, 224]
        feat2 = self.forward_once(x2_flat)  # [B*C, N, D]

        out = feat2.view(b, c, feat2.shape[1], -1)  # [B, C, N, D]
        return feat1, out
    
    

class ViTSiameseNetworkSmall(nn.Module):
    def __init__(self, pretrained_model='vit_small_patch16_224'):
        super(ViTSiameseNetworkSmall, self).__init__()
        self.vit = timm.create_model(pretrained_model, pretrained=True)
        self.vit.reset_classifier(0, '')
        self.log_sigma1 = nn.Parameter(torch.tensor(0.0))
        self.log_sigma2 = nn.Parameter(torch.tensor(0.0))

    def forward_once(self, x):
        outputs = self.vit.forward_features(x)
        return outputs

    def forward(self, x1, x2):
        feat1 = self.forward_once(x1)
        b,c = x2.shape[0], x2.shape[1]
        x2_flat = x2.view(b*c, x2.shape[2], x2.shape[3], x2.shape[4])
        feat2 = self.forward_once(x2_flat)
        out = feat2.view(b,c, 197, -1)

        return feat1, out

#Evaluation Dataset
class CSIIMDDataset(Dataset):
    def __init__(self, image_label_pairs, transform=None):
        """
        image_label_pairs: List of (image_path, label) pairs
        transform: Image transform (e.g., resize, normalize)
        """
        self.image_label_pairs = image_label_pairs
        self.transform = transform

    def __len__(self):
        return len(self.image_label_pairs)

    def __getitem__(self, idx):
        original_path, ID, impacts = self.image_label_pairs[idx]
        img1_path = f'YOUR_PATH_DATA/laion-5B/csi-imd/manipulated_images/{ID}_1.png'
        img2_path = f'YOUR_PATH_DATA/laion-5B/csi-imd/manipulated_images/{ID}_2.png'
        img3_path = f'YOUR_PATH_DATA/laion-5B/csi-imd/manipulated_images/{ID}_3.png'
        img4_path = f'YOUR_PATH_DATA/laion-5B/csi-imd/manipulated_images/{ID}_4.png'
        img5_path = f'YOUR_PATH_DATA/laion-5B/csi-imd/manipulated_images/{ID}_5.png'

        original = Image.open(original_path+'.jpg').convert("RGB")
        # original = resize_and_pad(original)

        img1 = Image.open(img1_path).convert("RGB")
        # img1 = resize_and_pad(img1)

        img2 = Image.open(img2_path).convert("RGB")
        # img2 = resize_and_pad(img2)

        img3 = Image.open(img3_path).convert("RGB")
        # img3 = resize_and_pad(img3)

        img4 = Image.open(img4_path).convert("RGB")
        # img4 = resize_and_pad(img4)

        img5 = Image.open(img5_path).convert("RGB")
        # img5 = resize_and_pad(img5)


        if self.transform:
            original = self.transform(original)
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)
            img4 = self.transform(img4)
            img5 = self.transform(img5)

        return original, img1, img2, img3, img4, img5, impacts
    
class RedditDataset(Dataset):
    def __init__(self, image_label_pairs, transform=None):
        """
        image_label_pairs: List of (image_path, label) pairs
        transform: Image transform (e.g., resize, normalize)
        """
        self.image_label_pairs = image_label_pairs
        self.transform = transform

    def __len__(self):
        return len(self.image_label_pairs)

    def __getitem__(self, idx):
        try:
            original_path, img1_path, img2_path = self.image_label_pairs[idx]

            original = Image.open(original_path.replace('/RedditProv', '/laion-5B/RedditProv_clean')).convert("RGB")
            # original = resize_and_pad(original)

            img1 = Image.open(img1_path.replace('/RedditProv', '/laion-5B/RedditProv_clean')).convert("RGB")
            # img1 = resize_and_pad(img1)

            img2 = Image.open(img2_path.replace('/RedditProv', '/laion-5B/RedditProv_clean')).convert("RGB")
            # img2 = resize_and_pad(img2)



            if self.transform:
                original = self.transform(original)
                img1 = self.transform(img1)
                img2 = self.transform(img2)


            return original, img1, img2, 1
        except:
            return torch.tensor([]), torch.tensor([]), torch.tensor([]), 0
    

def best_kendall_tau(predicted_ranking, possible_rankings):
    best_score = -1
    best_ranking = None

    for ranking in possible_rankings:
        tau, _ = kendalltau(predicted_ranking, ranking)
        if tau > best_score:
            best_score = tau
            best_ranking = ranking

    return best_ranking, best_score


def cosine_similarity(a, b):
    a = a.view(-1)
    b = b.view(-1)
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()

#################Load dataset ###################
csi_data = np.load('csi_imd_test_values.npy', allow_pickle=True)
csi_imd_test_set = CSIIMDDataset(csi_data, transform=transform)
csi_imd_loader = DataLoader(csi_imd_test_set, batch_size=1, shuffle=False)

data_Reddit_prov = np.load('YOUR_PATH_DATA//laion-5B/img_chains.npy')
reddit_dataset = RedditDataset(data_Reddit_prov, transform=transform)
reddit_loader = DataLoader(reddit_dataset, batch_size=1, shuffle=False)

data_easy_val = np.load('./data_selection/validation_easy.npy', allow_pickle=True)
data_hard_val = np.load('./data_selection/validation_hard.npy', allow_pickle=True)

val_dataset_easy= SiameseDataset(data_easy_val, transform=transform)
val_dataset_hard= SiameseDataset(data_hard_val, transform=transform)


val_loader_easy = DataLoader(val_dataset_easy, batch_size=1, shuffle=False)
val_loader_hard = DataLoader(val_dataset_hard, batch_size=1, shuffle=False)


#############Load model###############
model = ViTSiameseNetworkSmall().to("cuda")
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)


selection = '' #caption, image, all
ciriculum = '' #easy, hard

model.eval()

def score_order(scores, impacts):
    count = 0.0
    accuracy = 0.0
    for i in range(len(scores)):
        for j in range(i, len(scores)):
            if i == j:
                continue
            count +=1
            if scores[i] >= scores[j] and impacts[i] >= impacts[j]:
                accuracy += 1
            elif scores[i] <= scores[j] and impacts[i] <= impacts[j]:
                accuracy += 1
    return count, accuracy

try:
    path = 'YOUR MODEL PATH'
    model.load_state_dict(torch.load(path))
    print(f"Model loaded successfully! {path}")
except Exception as e:
    print(f"Error loading model: {e}")

scores = []
data_Reddit_prov = np.load('YOUR_PATH_DATA/laion-5B/img_chains.npy')
reddit_dataset = RedditDataset(data_Reddit_prov, transform=transform)
reddit_loader = DataLoader(reddit_dataset, batch_size=1, shuffle=False)
clean_reddit = 0
for original, img1, img2, found_flag in tqdm(reddit_loader):
    if found_flag == 0:
        continue
    else:
        clean_reddit += 1
    original_1, feat1 = model(original, img1.unsqueeze(0))
    original_1 = F.normalize(original_1.detach()[0], dim=-1)
    manip_1 = F.normalize(feat1.detach().detach()[0], dim=-1)
    score_1 = F.pairwise_distance(original_1, manip_1, p=2).mean()


    original_2, feat2 = model(original, img2.unsqueeze(0))
    original_2 = F.normalize(original_2.detach()[0], dim=-1)
    manip_2 = F.normalize(feat2.detach().detach()[0], dim=-1)
    score_2 = F.pairwise_distance(original_2, manip_2, p=2).mean()

    count, acc = score_order([score_1, score_2], [0,1])
    # print([score_1, score_2])
    scores.append((acc/count))

print(clean_reddit)
print(np.mean(scores))


distance = []
distance_1 = []
distance_2 = []
for original, manipulations, labels, amt, original_path, manipulations_path in tqdm(val_loader_easy):
    # original, manipulations = original.to('cuda'), manipulations.to('cuda')

    if amt > 1:
        # # total += 1
        original_1, feat = model(original, manipulations)

        original_1 = F.normalize(original_1.detach()[0], dim=-1)
        manip_1 = F.normalize(feat.detach()[0][0].detach()[0], dim=-1)
        manip_2 = F.normalize(feat.detach()[0][1].detach()[0], dim=-1)

        score_1 = F.pairwise_distance(original_1, manip_1, p=2).mean().cpu()
        score_2 = F.pairwise_distance(original_1, manip_2, p=2).mean().cpu()

        distance_1.append(score_1.detach().cpu().numpy())
        distance_2.append(score_2.detach().cpu().numpy())
        diff = score_1.cpu() - score_2.cpu()
        distance.append(diff.detach().cpu().numpy())

        np.save(f'distance_1_{ciriculum}_{selection}.npy', distance_1)
        np.save(f'distance_2_{ciriculum}_{selection}.npy', distance_2)
print(np.mean(distance))



accuracy = []
for original, img1, img2, img3, img4, img5, impacts in tqdm(csi_imd_loader):
    original, img1, img2, img3, img4, img5 = original.to('cuda'), img1.to('cuda'), img2.to('cuda'), img3.to('cuda'), img4.to('cuda'), img5.to('cuda')

    original_1, feat1 = model(original, img1.unsqueeze(0))
    original_1 = F.normalize(original_1, dim=-1)
    manip_1 = F.normalize(feat1.detach()[0], dim=-1)
    score_1 = F.pairwise_distance(original_1, manip_1, p=2).mean().item()

    original_2, feat2 = model(original, img2.unsqueeze(0))
    original_2 = F.normalize(original_2, dim=-1)
    manip_2 = F.normalize(feat2.detach()[0], dim=-1)
    score_2 = F.pairwise_distance(original_2, manip_2, p=2).mean().item()

    original_3, feat3 = model(original, img3.unsqueeze(0))
    original_3 = F.normalize(original_3, dim=-1)
    manip_3 = F.normalize(feat3.detach()[0], dim=-1)
    score_3 = F.pairwise_distance(original_3, manip_3, p=2).mean().item()

    original_4, feat4 = model(original, img4.unsqueeze(0))
    original_4 = F.normalize(original_4, dim=-1)
    manip_4 = F.normalize(feat4.detach()[0], dim=-1)
    score_4 = F.pairwise_distance(original_4, manip_4, p=2).mean().item()

    original_5, feat5 = model(original, img5.unsqueeze(0))
    original_5 = F.normalize(original_5, dim=-1)
    manip_5 = F.normalize(feat5.detach()[0], dim=-1)
    score_5 = F.pairwise_distance(original_5, manip_5, p=2).mean().item()

    scores = [score_1, score_2,score_3,score_4,score_5]

    # Mapping dictionary (handle both 'med' and 'medium')
    mapping = {'high': 3, 'medium': 2, 'med': 2, 'low': 1}

    # Flatten and map
    clean_array = [mapping[item[0].lower()] for item in impacts]

    count, acc = score_order(scores, clean_array)
    accuracy.append((acc/count))


    
print(np.mean(accuracy))

# vit_siamese_hard_1_small_caption_all_close_only_image_caption.pth
# vit_siamese_hard_1_small_caption_all_close_only_image.pth
# vit_siamese_hard_1_small_caption_all_close_only_no_class_head.pth
