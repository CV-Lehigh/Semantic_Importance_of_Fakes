import lpips
from skimage.metrics import structural_similarity as ssim
from skimage import io, color
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torch
from PIL import Image
import clip
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
from sklearn.linear_model import LinearRegression


# Load LPIPS model (AlexNet by default, but can use VGG or SqueezeNet)
loss_fn = lpips.LPIPS(net='alex')  # Options: 'alex', 'vgg', 'squeeze'

# Load CLIP 
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

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

def normalized_euclidean_distance_torch(A, B):
    A_normalized = A / torch.norm(A)
    B_normalized = B / torch.norm(B)
    return torch.norm(A_normalized - B_normalized)

def cosine_distance(a, b):
    A_normalized = a / torch.norm(a)
    B_normalized = b / torch.norm(b)
    a_flat = a.view(A_normalized.shape[0], -1)  # Flatten to (batch, features)
    b_flat = b.view(B_normalized.shape[0], -1)
    return 1 - torch.nn.functional.cosine_similarity(a_flat, b_flat, dim=-1)

def plot_new_data(sim_data, split, selection):
    # Plot
    colours = ['red','green', 'purple', 'blue', 'orange', 'brown']
    if selection == 'image_':
        labels = ['close sphere','close PAID']
        save_figure = ['image_close_sphere', 'image_close_PAID']
    else: 
        labels = ['close sphere','close linear','close PAID']
        save_figure = ['caption_close_sphere', 'caption_close_linear', 'caption_close_PAID']
    # try:
    np_array = np.array(sim_data[split])
    x = np_array[:,0]
    y = np_array[:,1]

    plt.scatter(x, y, color=colours[split], label=labels[split], marker='o')
    plt.title('Movement in LPIPS and CLIP Score')
    plt.xlabel('LPIPS Distance')
    plt.ylabel('CLIP Score')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{save_figure[split]}.png')
    plt.clf()


different_interpolations = [['image_prompt_close_sphere_768','image_prompt_close_PAID'],  ['prompt_close_sphere_768', 'prompt_close_linear_768', 'prompt_close_PAID_0']]
for index in range(len(different_interpolations)):
    similarity_data = [[],[],[]]
    file_paths = [[],[],[]]

    interp_method = 0
    update = 0
    interpolation_set = different_interpolations[index]
    if index == 0:
        selection = 'image_'
    else:
        selection == 'caption_'
    for interpolation in interpolation_set:
        #YOUR_PATH_HERE
        folder_path = f'YOUR_PATH_HERE/laion-5B/{interpolation}'
        ext = ''
        start = '0'
        emb_str = 'embedding'
        if 'PAID' not in interpolation:
            ext = 'interpolation_lstitms-0_1_step-'
            start = 'start'
            emb_str = 'embeddings'
        print(f"now working on: {interpolation}")
        for experiment in tqdm(os.listdir(f'{folder_path}')):
            try:
                img1_path = f"{folder_path}/{experiment}/images/{ext}{start}.png"
                emb_1_path = f"{folder_path}/{experiment}/{emb_str}/{ext}{start}.pt"  
                num_files = len(os.listdir(f'{folder_path}/{experiment}/images/'))

                for idx in range(1,(num_files-2)):
                    img2_path = f"{folder_path}/{experiment}/images/{ext}{idx}.png"
                    emb_2_path = f"{folder_path}/{experiment}/{emb_str}/{ext}{idx}.pt"

                    # Convert images to tensors
                    img1 = load_image(img1_path)
                    img2 = load_image(img2_path)

                    #Get embeddings
                    if 'PAID' not in interpolation:
                        emb1 = torch.load(emb_1_path)['image_embed']
                        emb2 = torch.load(emb_2_path)['image_embed']
                    else:
                        emb1 = torch.load(emb_1_path)
                        emb2 = torch.load(emb_2_path)

                    euclid_dist = normalized_euclidean_distance_torch(emb1, emb2).item()


                    # Compute image space
                    lpips_distance = loss_fn(img1, img2).item()
                    clip_sim = clip_image_similarity(img1_path, img2_path)
                    score_clip = max(100*clip_sim,0)
                    similarity_data[interp_method].append([lpips_distance, score_clip, euclid_dist])
                    file_paths[interp_method].append([img1_path, img2_path])
                    
                    # Save
                    np.save(f'{selection}/{selection}sim_data_final.npy', np.array(similarity_data, dtype=object))
                    np.save(f'{selection}/{selection}file_paths_final.npy', np.array(file_paths, dtype=object))
                if update % 25 == 0:
                    plot_new_data(similarity_data, interp_method, selection)
                update += 1
            except:
                print(f'Failed on experiment {experiment}')
        interp_method +=1


