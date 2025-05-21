## 1. Data Generation

### Pre-defined **TO DO**
1. Download the set of increasing manipulations using this link <> and place it into your ./data folder

### Custom Generation
1. Download the embeddings zip folder https://drive.google.com/file/d/18SEK2DObfhH4U9Bxw9Qb2DH3xRGy5Q_k/view?usp=sharing and place it into your './data/laion-5B' folder -- this file provides the latent noise embeddings for image + prompt interpolation
2. Download and set up the evnironment for the Linear and Spherical interpolation github project: https://github.com/alen-smajic/Stable-Diffusion-Latent-Space-Explorer/tree/main?tab=readme-ov-file and the PAID interpolation project: https://github.com/QY-H00/attention-interpolation-diffusion

    2.1:  Copy and paste the files in replace-latent-space-explorer in to the Stable-Diffusion-Latent-Space project and run ```run_interpolation_experiment.py``` (note: For generating linear prompt interpolation change config file 'prompt_interpolation.yaml': change 'output_path' and set 'interpolation method' to lerp. Consider using nohup this takes about 24 hours.

    2.2:  Copy and past the files in replace_attention_diffusion in to the attention-interpolation-diffusion project and run ```run_interpolation_experiment.py```. Consider using nohup this takes about 24 hours.


At this point you should have in your ./data/laion-5B folder 5 newly greated and populated folders image_prompt_close_PAID, image_prompt_close_sphere_768, prompt_close_PAID, prompt_close_sphere_768, prompt_close_linear_768

### Data Fitting and labeling
1. Navigate to the Semantic_Importance_of_Fakes folder
2. ```conda env create -f environment.yaml``` (python==3.10.16)
3. ```conda activate siof```
4. Run ```nohup python plot.py > plot.log 2&>1 &``` generates LPIPS and CLIP-Scores for each pair (original image, manipulated image)
5. Run ```python fit.py``` finds the best fit curves, and defines the hard case and easy case parameters discussed in section 3.2 of the paper.
6. Run ```python select_images.py``` labels each data pair (original image, manipulated image) and generates the required triplets (original image, semantically matching manipulated image, semantically non-matching manipulated image) -- note this applyies the probabilistic pseudo-labeling so each time this is run the labels of the hard cases will change.

## 2. Training
1. Run ```nohup python ViT_siamese.py > training_log 2&>1 &``` Trains the easy case initially
2. Cmment out easy case training and uncomment the hard case then run the same code again
