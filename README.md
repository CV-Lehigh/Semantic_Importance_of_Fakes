# Semantic Importance of Fakes

> **NOTE:** Wherever you see `YOUR DATA PATH`, replace it with the path to your `./data/` directory.

---

## 1. Data Generation

### ✅ Pre-defined Generation

1. Download the set of increasing manipulations from **[this link](<>)** and place the contents into your `./data/` folder.

---

### 🛠️ Custom Generation

1. Download the **embeddings zip** file from  
   [Google Drive](https://drive.google.com/file/d/18SEK2DObfhH4U9Bxw9Qb2DH3xRGy5Q_k/view?usp=sharing)  
   and place it into the `./data/laion-5B/` directory.  
   This file provides the latent noise embeddings for image + prompt interpolation.

2. Set up the interpolation tools:
   - **Linear and Spherical Interpolation Project**  
     [Stable-Diffusion-Latent-Space-Explorer](https://github.com/alen-smajic/Stable-Diffusion-Latent-Space-Explorer)
   - **PAID Interpolation Project**  
     [attention-interpolation-diffusion](https://github.com/QY-H00/attention-interpolation-diffusion)

#### 🔄 Run the interpolation scripts

- **2.1 Stable-Diffusion-Latent-Space:**
  - Copy the contents of `replace-latent-space-explorer/` into the project folder Stable-Diffusion-Latent-Space-Explorer.
  - cd to Stable-Diffusion-Latent-Space-Explorer
  - Run the interpolation:
    ```bash
    nohup python run_interpolation_experiment.py &
    ```
    > *For linear prompt interpolation, edit `prompt_interpolation.yaml`: set `output_path` and change `interpolation method` to `lerp`. This may take ~24 hours.*

- **2.2 PAID Interpolation:**
  - Copy the contents of `replace_attention_diffusion/` into the project folder attention-interpolation-diffusion.
  - cd to attention-interpolation-diffusion
  - Run:
    ```bash
    nohup python run_interpolation_experiment.py &
    ```

Once complete, your `./data/laion-5B/` directory should contain:

- `image_prompt_close_PAID/`  
- `image_prompt_close_sphere_768/`  
- `prompt_close_PAID_0/`  
- `prompt_close_sphere_768/`  
- `prompt_close_linear_768/`

---

### 📈 Data Fitting and Labeling

1. Navigate to the `Semantic_Importance_of_Fakes/` directory.
2. Create the Conda environment:
   ```bash
   conda env create -f environment.yaml  # (Python 3.10.16)
   conda activate siof
   
3. Run:
    ```bash
   nohup python plot.py > plot.log 2&>1 &
generates LPIPS and CLIP-Scores for each pair (original image, manipulated image)

4. Run: `bash python fit.py` finds the best fit curves, and defines the hard case and easy case parameters discussed in section 3.2 of the paper.
   
5. Run ```python select_images.py``` labels each data pair (original image, manipulated image) and generates the required triplets (original image, semantically matching manipulated image, semantically non-matching manipulated image) -- note this applyies the probabilistic pseudo-labeling so each time this is run the labels of the hard cases will change.
   
   5.1 Between lines 43 - 57 decide which data split to run (all, image, caption)

## 2. Training
1. Run ```nohup python ViT_siamese.py > training_log 2&>1 &``` Trains the easy case initially -- Fill in line 377 with desred selection (caption, image, all)
2. Comment out easy case training and uncomment the hard case then run the same code again -- Place best model path in line 418

## 3. Evaluate
1. Run ``` python evalaute.py``` -- fill in lines 358 and 359 with desired selection (caption, image, all) and cirriculum (easy or hard)
    1.1 You can use the pretrained models avaoilable here: [Pre-trained_weights](https://drive.google.com/drive/folders/1xr4T_7dXJ3LV_zumrI08SxdwoycRVDSn?usp=sharing)
