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
import timm
import torchvision.models as models
from torchvision import transforms
from torchvision.transforms import InterpolationMode

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# google/vit-small-patch16-224
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
    random.shuffle(all_samples)

    # compute split point at 85%
    split_idx = int(ratio * len(all_samples))

    train_samples = all_samples[:split_idx]
    val_samples   = all_samples[split_idx:]
    return train_samples, val_samples


def list_to_tuple(x):
    return tuple(list_to_tuple(i) if isinstance(i, list) else i for i in x)

def split_sets(array1, array2):
    # Convert lists to tuples for hashable comparison
    set_array2 = set(list_to_tuple(x) for x in array2)

    # Filter matching and non-matching
    matching = np.array([x for x in array1 if list_to_tuple(x) in set_array2], dtype=object)
    non_matching = np.array([x for x in array1 if list_to_tuple(x) not in set_array2], dtype=object)
    return non_matching, matching

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
            return original, manipulations_torch, torch.tensor(labels, dtype=torch.float32), prev_amount
        except:
            print(f"failed on image: {self.authentic_images[idx]} with images {self.manipulated_images[idx]}")

###Model
class ViTSiameseNetwork(nn.Module):
    def __init__(self, pretrained_model='google/vit-base-patch16-224', dropping = True):
        super(ViTSiameseNetwork, self).__init__()
        if dropping == True:
            config = ViTConfig(
                drop_path_rate=0.1  # stochastic depth rate
            )

            # Load model with modified config
            self.vit = ViTModel(config=config)
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

class ResNetSiameseNetwork(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNetSiameseNetwork, self).__init__()

        # Load pretrained ResNet-101 and remove the classification head (fc layer)
        resnet = models.resnet101(pretrained=False)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])  # Keep up to conv5_x
        self.embedding_dim = 2048  # Output channels of last conv layer in ResNet101

    def forward_once(self, x):
        features = self.backbone(x)  # Output shape: (B, 2048, H/32, W/32)
        return features

    def forward(self, x1, x2):
        feat1 = self.forward_once(x1)  # shape: (B, 2048, H/32, W/32)

        # Flatten x2 to run as one batch
        b, m = x2.shape[0], x2.shape[1]
        x2_flat = x2.view(b * m, x2.shape[2], x2.shape[3], x2.shape[4])

        feat2 = self.forward_once(x2_flat)
        feat2 = feat2.view(b, m, self.embedding_dim, feat2.shape[2], feat2.shape[3])  # (B, C, 2048, H/32, W/32)

        return feat1, feat2


class ViTSiameseNetworkSmall(nn.Module):
    def __init__(self, _model='vit_small_patch16_224'):
        super(ViTSiameseNetworkSmall, self).__init__()
        self.vit = timm.create_model(_model, pretrained=True)
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
    
####DiNO Model
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
    
def contrastive_loss_func(anchor, pair, label, margin=.5):
    anchor = F.normalize(anchor, dim=-1) 
    pair = F.normalize(pair, dim=-1)
    d = F.pairwise_distance(anchor, pair, p=2)
    loss = (1 - label) * d**2 + label * F.relu(margin - d)**2
    return loss.mean()


def single_class_loss(anchor, values, label, temperature=0.2, margin=0.2, top_k=6):
    # Use CLS token hopefully better semantic features
    anchor_feat = F.normalize(anchor[:, 0], dim=-1)  # [1, 768]

    if label == 1:
        negatives_feat = F.normalize(values[:, 0], dim=-1)  # [x, 768]
        similarities = torch.matmul(negatives_feat, anchor_feat.T).squeeze(1)  # [x]
        topk_vals, _ = similarities.topk(k=min(top_k, negatives_feat.size(0)), largest=True)
        loss = F.relu(topk_vals - margin).mean()
        return loss

    elif label == 0:
        positive_feat = F.normalize(values[:, 0], dim=-1)  # [x, 768]
        logits = torch.matmul(anchor_feat, positive_feat.T) / temperature  # [1, x]
        targets = torch.tensor(0, device=logits.device)
        loss = F.cross_entropy(logits, targets.unsqueeze(0))
        return loss
    


def triplet_loss(anchor, values, labels, margin=1.0):
    positives = []
    negatives = []

    for index in range(len(labels)):
        if labels[index] == 0:
            positives.append(values[index])
        elif labels[index] == 1:
            negatives.append(values[index])
    

    #all posibile triplet combinations
    anchor = F.normalize(anchor[:, 1:], dim=-1)       # [1, 197, 768]
    positives = F.normalize(torch.stack(positives)[:, 1:], dim=-1) # [1, 197, 768]
    negatives = F.normalize(torch.stack(negatives)[:, 1:], dim=-1) # [1, 197, 768]
    d_ap = F.pairwise_distance(anchor, positives, p=2)
    d_an = F.pairwise_distance(anchor, negatives, p=2)
    loss = F.relu(d_ap - d_an + margin)
    return loss.mean()


def train(model, loader, val_loader, optimizer, criterion_trip, epochs=10, ciriculum = 'easy', selection = 'caption'):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        model.train()
        for original, manipulations, labels, amt in tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}"):
            original, manipulations, labels, amt = original.to(device), manipulations.to(device), labels.to(device).unsqueeze(1), amt.to(device)
            optimizer.zero_grad()
            original_outputs, manipulated_outputs = model(original, manipulations)

            triplet_loss_ = torch.tensor(0.0).to(device)
            triple_count = 0

            for index in range(original_outputs.shape[0]):
                original_features = original_outputs[index].unsqueeze(0)
                manipulated_features = manipulated_outputs[index][:amt[index]]
                itm_labels = labels[index][0][:amt[index]]
                if amt[index] <= 1:
                    continue

                triplet_loss_ = triplet_loss_ + criterion_trip(original_features, manipulated_features, itm_labels)
                triple_count += 1
            
            loss = ((triplet_loss_)/triple_count)
            loss = loss.to(device)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(loader)
    

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for img1, manipulations, labels, amt in tqdm(val_loader):
                img1, manipulations, labels, amt = img1.cuda(), manipulations.cuda(), labels.cuda().unsqueeze(1), amt
                original_outputs, manipulated_outputs = model(img1, manipulations)
                triplet_loss_value_eval = 0
                triple_count = 0
                for index in range(original_outputs.shape[0]):
                    original_features = original_outputs[index].unsqueeze(0)
                    manipulated_features = manipulated_outputs[index][:amt[index]]
                    itm_labels = labels[index][0][:amt[index]]
                    if amt[index] <= 1:
                        continue

                    triplet_loss_value_eval = triplet_loss_value_eval + criterion_trip(original_features, manipulated_features, itm_labels)
                    triple_count += 1
                val_loss += ((triplet_loss_value_eval)/triple_count)


            avg_val_loss = val_loss / len(val_loader)

        if epoch == 0 or best_loss > val_loss:
            best_loss = val_loss
        # Your data Path #
        save_path = f'YOUR_DATA_PATH/laion-5B/model_paths/vit_siamese_{ciriculum}_{epoch}_small_{selection}.pth'
        torch.save(model.state_dict(), save_path)
        print(save_path)

        print(f"Epoch {epoch+1} Summary: ")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss:   {avg_val_loss:.4f}")



############################## Easy case training  #####################
print('############################## Easy case training  #####################')
selection = '' #caption, image, all

model = ViTSiameseNetworkSmall().to(device)
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)


criterion_trip = triplet_loss
contrastive_loss = contrastive_loss_func
optimizer = AdamW(model.parameters(), lr=3e-5, weight_decay=1e-2)
best_loss = float('inf')

easy_val_set = np.load('./data_selection/validation_easy_data.npy', allow_pickle=True)
hard_val_set = np.load('./data_selection/validation_hard_data.npy',  allow_pickle=True)

data_easy = np.load(f'./data_selection/1std_{selection}_easy_cases_cnt_random_trip_close_only_data.npy', allow_pickle=True)
data_easy_train, data_easy_val = generate_splits(data_easy)


train_dataset_easy = SiameseDataset(data_easy_train, transform=transform)
val_dataset_easy = SiameseDataset(data_easy_val, transform=transform)

train_loader_easy = DataLoader(train_dataset_easy, batch_size=128, shuffle=True)
val_loader_easy = DataLoader(val_dataset_easy, batch_size=128, shuffle=False)

train(model, train_loader_easy, val_loader_easy, optimizer, criterion_trip, contrastive_loss, epochs=3, selection = selection)


################### Find the best easy model then Run Hard set #################################################################
############################# Hard case training  #####################
# print('############################## Hard case training  #####################')
# data_hard = np.load('./data_selection/1std_image_hard_cases_cnt_random_trip_close_only.npy', allow_pickle=True)
# data_hard_train, data_hard_val = generate_splits(data_hard)

# np.save('./data_selection/image_validation_hard.npy', data_hard_val)


# train_dataset_hard = SiameseDataset(data_hard_train, transform=transform)
# val_dataset_hard = SiameseDataset(data_hard_val, transform=transform)

# train_loader_hard = DataLoader(train_dataset_hard, batch_size=128, shuffle=True)
# val_loader_hard = DataLoader(val_dataset_hard, batch_size=128, shuffle=False)

# try:
#     model.load_state_dict(torch.load('../../../data/jpk322/laion-5B/model_paths/vit_siamese_easy_1_small_caption_all_close_only_image.pth'))
#     print("Model loaded successfully!")

#     train(model, train_loader_hard, val_loader_hard, optimizer, criterion_trip, epochs=2, ciriculum='hard', selection=selection)
# except Exception as e:
#     print(f"Error loading model: {e}")