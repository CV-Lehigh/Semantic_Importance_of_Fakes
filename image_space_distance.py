import lpips
from skimage.metrics import structural_similarity as ssim
from skimage import io, color
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from scipy.spatial.distance import euclidean
import clip
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from sentence_transformers import SentenceTransformer, util
import matplotlib.pyplot as plt
from torchmetrics.multimodal.clip_score import CLIPScore
import numpy as np
import os
from tqdm import tqdm
import pandas as pd
import cv2
from transformers import BlipProcessor, BlipForConditionalGeneration
from nltk.translate.meteor_score import meteor_score
import math
from nltk.tokenize import word_tokenize
import nltk
import json


 # Ensure NLTK's tokenizer is available
nltk.download("punkt") 
nltk.download('punkt_tab')  
nltk.download('wordnet')

with open("./clipscore/scores.json", "r") as json_file:
    scores = json.load(json_file)

# Load LPIPS model (AlexNet by default, but can use VGG or SqueezeNet)
loss_fn = lpips.LPIPS(net='alex')  # Options: 'alex', 'vgg', 'squeeze'

# Load CLIP 
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")  # or "Salesforce/blip2-flan-t5-xl"
BLIP_caption = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)
BLIP_caption.to(device)

def clip_image_similarity(image1_path, image2_path):
    """
    Compute the CLIP similarity score between two images.

    Args:
        image1_path (str): Path to the first image.
        image2_path (str): Path to the second image.

    Returns:
        float: Cosine similarity score (higher means more similar).
    """
    # Load and preprocess images
    image1 = preprocess(Image.open(image1_path)).unsqueeze(0).to(device)
    image2 = preprocess(Image.open(image2_path)).unsqueeze(0).to(device)

    # Encode images
    with torch.no_grad():
        image1_features = model.encode_image(image1).float()
        image2_features = model.encode_image(image2).float()

    # Normalize and compute cosine similarity
    
    image1_features /= image1_features.norm(dim=-1, keepdim=True)
    image2_features /= image2_features.norm(dim=-1, keepdim=True)

    similarity = (image1_features @ image2_features.T).item()

    return similarity


# Load and preprocess images
def load_image(image_path, use='LPIPS'):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize to 256x256 for LPIPS
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
    ])
    if use == 'CLIPScore':
        transform = transforms.Compose([
            transforms.ToTensor(),  # Convert to tensor
        ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)  # Add batch dimension

def compute_grid_dimensions(x):
    if x <= 0:
        return (0, 0)
    
    cols = math.ceil(math.sqrt(x))  # Start with square-like columns
    rows = math.floor(x / cols)  # Compute rows based on columns
    
    if rows * cols < x:  # If we don't fit all images, increase rows
        rows += 1
    
    return rows, cols

def plot(plots, plot_titles, dimension_x, dimension_y, save_name, title):
    x = 5*len(plots)
    plt.figure(figsize=(15, 6))
    plt.suptitle(title, fontsize=12)


    for idx, plot in enumerate(plots):
        plt.subplot(dimension_y, dimension_x, (idx+1))
        plt.title(f"{plot_titles[idx]}")
        plt.imshow(plot)
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(save_name, bbox_inches='tight', pad_inches=.1)

def normalized_euclidean_distance_torch(A, B):
    A_normalized = A / torch.norm(A)
    B_normalized = B / torch.norm(B)
    return torch.norm(A_normalized - B_normalized)


def generate_caption(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(BLIP_caption.device)
    caption_ids = BLIP_caption.generate(**inputs)
    caption = processor.batch_decode(caption_ids, skip_special_tokens=True)[0]
    return caption

def compute_meteor(generated_caption, reference_captions):
    # Tokenize the input captions
    generated_tokens = word_tokenize(generated_caption)
    reference_tokens = [word_tokenize(ref) for ref in reference_captions]
    
    # Compute METEOR score
    return meteor_score(reference_tokens, generated_tokens)

###Evaluation with Cosine Distancing
def cosine_distance(a, b):
    A_normalized = a / torch.norm(a)
    B_normalized = b / torch.norm(b)
    a_flat = a.view(A_normalized.shape[0], -1)  # Flatten to (batch, features)
    b_flat = b.view(B_normalized.shape[0], -1)
    return 1 - torch.nn.functional.cosine_similarity(a_flat, b_flat, dim=-1)



LPIPS_all = []
EUc_all= []
CLIP_sim_all = []
CAP_sim_all = []
cos_all = []
CLIP_score_all = []
folder_path = '../../../data/jpk322/laion-5B/prompt_close_PAID'

###How to average across many many experiments????
captions = np.load('../Stable-Diffusion-Latent-Space-Explorer/captions_most_alike.npy')


i = 0
for experiment in tqdm(os.listdir(f'{folder_path}')):
    # if '7727' not in experiment:
    #     continue
    try:
        LPIPS = []
        EUc= []
        cos = []
        CLIP_sim = []
        CLIP_scores = []
        CAP_sim = []
        # Paths to the images
        img1_path = f"{folder_path}/{experiment}/images/0.png"
        emb_1_path = f"{folder_path}/{experiment}/embedding/0.pt"
        original_caption = generate_caption(img1_path)
        num_files = len(os.listdir(f'{folder_path}/{experiment}/images/'))

        splits = experiment.split("_")
        caption_row = captions[:, 0] == splits[0]+'.jpg' 
        num_files = len(os.listdir(f'{folder_path}/{experiment}/images/'))
        names = ['start']
        img_list = []
        img_list.append(cv2.cvtColor(cv2.imread(f'{folder_path}/{experiment}/images/0.png'),cv2.COLOR_BGR2RGB))

        for idx in range(1,(num_files-1)):
                img2_path = f"{folder_path}/{experiment}/images/{idx}.png"
                emb_2_path = f"{folder_path}/{experiment}/embedding/{idx}.pt"

                # Convert images to tensors
                img1 = load_image(img1_path)
                img2 = load_image(img2_path)

                #Get embeddings
                # emb1 = torch.load(emb_1_path)['image_embed']
                # emb2 = torch.load(emb_2_path)['image_embed']
                emb1 = torch.load(emb_1_path)
                emb2 = torch.load(emb_2_path)

                # Compute image space
                lpips_distance = loss_fn(img1, img2).item()
                LPIPS.append(lpips_distance)
                # print(f"  LPIPS Distance: {lpips_distance.item():.4f}") 


                #compute semantic space
                euclid_dist = normalized_euclidean_distance_torch(emb1, emb2).item()
                EUc.append(euclid_dist)

                cos_dist = cosine_distance(emb1,emb2).item()
                cos.append(cos_dist)

                clip_sim = clip_image_similarity(img1_path, img2_path)
                CLIP_sim.append(clip_sim)

                generated_caption = generate_caption(img2_path)
                meteor_sim = compute_meteor(generated_caption, original_caption) 
                CAP_sim.append(meteor_sim)

                score_clip = max(100*clip_sim,0)
                CLIP_scores.append(score_clip)

                # img_list.append(cv2.cvtColor(cv2.imread(f'{folder_path}/{experiment}/images/interpolation_lstitms-0_1_step-{idx}.png'),cv2.COLOR_BGR2RGB))
                # names.append(f"{idx} -- Euc: {euclid_dist:.3f} LPIPS: {lpips_distance:.3f} CLIP:{score_clip:.3f}")
    except:
        print(f"failed {experiment}")
        
    LPIPS_all.append(LPIPS)
    EUc_all.append(EUc)
    CLIP_sim_all.append(CLIP_sim)
    CAP_sim_all.append(CAP_sim)
    cos_all.append(cos)
    CLIP_score_all.append(CLIP_scores)

    
    # rows, col = compute_grid_dimensions((num_files-1))
    # title = f"{captions[caption_row][0][1]} --> {captions[caption_row][0][3]}: {captions[caption_row][0][4]}"
    # print(experiment)
    # plot(img_list, names,rows,col, f'viz/{experiment}.png', title)
    # plt.clf()


    # CAP_sim_all.append(CAP_sim)
    if i % 10 == 0 or i == 3479:
        #Means for metrics
        # print(f"Saving: {i}")
        df_LPIPS = pd.DataFrame([row + [None] * (max(map(len, LPIPS_all)) - len(row)) for row in LPIPS_all])
        column_means_LPIPS = df_LPIPS.mean(skipna=True)

        df_eu = pd.DataFrame([row + [None] * (max(map(len, EUc_all)) - len(row)) for row in EUc_all])
        column_means_eu = df_eu.mean(skipna=True)

        df_clip = pd.DataFrame([row + [None] * (max(map(len, CLIP_sim_all)) - len(row)) for row in CLIP_sim_all])
        column_means_clip = df_clip.mean(skipna=True)

        df_meteor = pd.DataFrame([row + [None] * (max(map(len, CAP_sim_all)) - len(row)) for row in CAP_sim_all])
        column_means_meteor = df_meteor.mean(skipna=True)

        df_cosine = pd.DataFrame([row + [None] * (max(map(len, cos_all)) - len(row)) for row in cos_all])
        column_means_cosine = df_cosine.mean(skipna=True)

        df_clipS = pd.DataFrame([row + [None] * (max(map(len, CLIP_score_all)) - len(row)) for row in CLIP_score_all])
        column_means_clipS = df_clipS.mean(skipna=True)



        plt.plot(column_means_eu, column_means_LPIPS, marker='o', linestyle='-')

        # Labels and title
        plt.xlabel("Euclidean Distance")
        plt.ylabel("LPIPS")
        plt.title("Effect movement in SD space has on Strctural representation")

        plt.savefig("viz/LPIPS.png", dpi=300, bbox_inches='tight')
        plt.clf()



        plt.plot(column_means_cosine, column_means_LPIPS, marker='o', linestyle='-')

        plt.xlabel("Cosine Distance")
        plt.ylabel("LPIPS")
        plt.title("Effect movement in SD space has on Strctural representation")

        plt.savefig("viz/Cosine_LPIPS.png", dpi=300, bbox_inches='tight')
        plt.clf()



        plt.plot(column_means_eu, column_means_clip, marker='o', linestyle='-')

        # Labels and title
        plt.xlabel("Euclidean Distance")
        plt.ylabel("CLIP Similarity")
        plt.title("Effect movement in SD space has on CLIP representation")

        plt.savefig("viz/CLIP.png", dpi=300, bbox_inches='tight')
        plt.clf()

        
        plt.plot(column_means_cosine, column_means_clip, marker='o', linestyle='-')

        # Labels and title
        plt.xlabel("Cosine Distance")
        plt.ylabel("CLIP Similarity")
        plt.title("Effect movement in SD space has on CLIP representation")

        plt.savefig("viz/Cosine_CLIP.png", dpi=300, bbox_inches='tight')
        plt.clf()


        plt.plot(column_means_eu, column_means_clipS, marker='o', linestyle='-')

        # Labels and title
        plt.xlabel("Euclidean Distance")
        plt.ylabel("CLIP Score")
        plt.title("Effect movement in SD space has on CLIP Score")

        plt.savefig("viz/CLIP_score.png", dpi=300, bbox_inches='tight')
        plt.clf()

        
        plt.plot(column_means_cosine, column_means_clipS, marker='o', linestyle='-')

        # Labels and title
        plt.xlabel("Cosine Distance")
        plt.ylabel("CLIP Score")
        plt.title("Effect movement in SD space has on CLIP Score")

        plt.savefig("viz/Cosine_CLIP_score.png", dpi=300, bbox_inches='tight')
        plt.clf()


        plt.plot(column_means_eu, column_means_meteor, marker='o', linestyle='-')

        # Labels and title
        plt.xlabel("Euclidean Distance")
        plt.ylabel("METEOR")
        plt.title("Effect movement in SD space has on Caption representation")

        plt.savefig("viz/METEOR.png", dpi=300, bbox_inches='tight')
        plt.clf()


        plt.plot(column_means_cosine, column_means_meteor, marker='o', linestyle='-')

        # Labels and title
        plt.xlabel("Cosine Distance")
        plt.ylabel("METEOR")
        plt.title("Effect movement in SD space has on Caption representation")

        plt.savefig("viz/Cosine_METEOR.png", dpi=300, bbox_inches='tight')
        plt.clf()




        # plt.plot(EUc, CAP_sim, marker='o', linestyle='-')

        # # Labels and title
        # plt.xlabel("Euclidean Distance")
        # plt.ylabel("BERT Similarity")
        # plt.title("Effect movement in SD space has on Caption representation")

        # plt.savefig("BERT_all.png", dpi=300, bbox_inches='tight')
        # plt.clf()



        plt.plot(column_means_LPIPS, column_means_clip, marker='o', linestyle='-')

        # Labels and title
        plt.xlabel("LPIPS")
        plt.ylabel("CLIP Similarity")
        plt.title("Effect structural change has on CLIP representation")

        plt.savefig("viz/CLIP_LPIPS.png", dpi=300, bbox_inches='tight')
        plt.clf()


        plt.plot(column_means_LPIPS, column_means_clipS, marker='o', linestyle='-')

        # Labels and title
        plt.xlabel("LPIPS")
        plt.ylabel("CLIP Score")
        plt.title("Effect structural change has on CLIP Score")

        plt.savefig("viz/CLIP_score_LPIPS.png", dpi=300, bbox_inches='tight')
        plt.clf()



        plt.plot(column_means_LPIPS, column_means_meteor, marker='o', linestyle='-')

        # Labels and title
        plt.xlabel("Euclidean Distance")
        plt.ylabel("METEOR")
        plt.title("Effect structural changehas on Caption representation")

        plt.savefig("viz/LPIPS_METEOR.png", dpi=300, bbox_inches='tight')
        plt.clf()



        # plt.plot(LPIPS, CAP_sim, marker='o', linestyle='-')

        # # Labels and title
        # plt.xlabel("LPIPS")
        # plt.ylabel("BERT Similarity")
        # plt.title("Effect movement structural has on BERT representation")

        # plt.savefig("BERT_LPIPS_all.png", dpi=300, bbox_inches='tight')
        # plt.clf()
    i+=1
