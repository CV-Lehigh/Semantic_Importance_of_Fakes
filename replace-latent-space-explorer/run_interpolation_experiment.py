import run_sd_experiment  # Runs script.py
import numpy as np
import runpy
import sys
import subprocess
from tqdm import tqdm
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed


rows = np.load('captions_alike.npy')

def assemble_args(data, use_image=False):
    if use_image == True:
        return  [
            "--exp_config", "./configs/image_prompt_interpolation.yaml",
            "--prompt_in", f"\"{remove_apostrophes_and_quotes(data[1])}|low resolution, blurry, random, cartoon\"",
            "--prompt_out", f"\"{remove_apostrophes_and_quotes(data[3])}|low resolution, blurry, random, cartoon\"",
            "--latent_in", f"\"YOUR DATA PATH/img_emb/{data[0][:-4]}_img2img_single-inference/embeddings/output-0_diffstep-25.pt\" "
            "--latent_out", f"\"YOUR DATA PATH/laion-5B/img_emb/{data[2][:-4]}_img2img_single-inference/embeddings/output-0_diffstep-25.pt\" "
            "--idx", data[0][:-4],
            "--num_steps", '10'
        ]
    return  [
        "--exp_config", "./configs/prompt_interpolation.yaml",
        "--prompt_in", f"\"{remove_apostrophes_and_quotes(data[1])}|low resolution, blurry, random, cartoon\"",
        "--prompt_out", f"\"{remove_apostrophes_and_quotes(data[3])}|low resolution, blurry, random, cartoon\"",
        "--idx", data[0][:-4],
        "--num_steps", '10'
    ]

def remove_apostrophes_and_quotes(text):
  """Removes all apostrophes and quote marks from a string.

  Args:
    text: The input string.

  Returns:
    The string with all apostrophes and quote marks removed.
  """
  text = text.replace("'", "\'")
  text = text.replace('"', "\"")
  return text


#0.005 0.01, 0.025, 0.05
#Resulting ~approx. steps: 50, 30, 15, 5
diff_per_step = 0.025  #should be a low modular value (0.0005-0.05)
def get_steps(difference):
    sim_gap = 1.0 - float(difference) 
    num_steps = int(sim_gap / diff_per_step)
    return num_steps

def run_task(command):
    subprocess.Popen(command, shell=True)


# for data in tqdm(range(0, len(rows), 4)):
    # Task definition
def run_on_gpu(gpu_id, data, use_image=False):
    script = "/home/jpk322/Stable-Diffusion-Latent-Space-Explorer/run_sd_experiment.py"

    # Corrected nohup command wrapped in bash -c
    args = assemble_args(data, use_image)
    command = f"nohup bash -c 'CUDA_VISIBLE_DEVICES={gpu_id} python {script} {' '.join(args)}'"
    print(command)


    # Create threads to run each process in parallel
    return command


def main():
    print("Starting job on 4 GPUs...")
    
    # Start the processes for each GPU (assuming you have 4 GPUs)
    for data in tqdm(range(0, len(rows)-4, 4)):
        processes = []
        for idx in range(1,4):
            # Run the task for each GPU
            save_folder = f'YOUR DATA PATH/laion-5B/image_prompt_close_sphere_768/{rows[data+idx][0][:-4]}_txt2img_interpolation'
            save_folder_1 = f'YOUR DATA PATH/laion-5B/prompt_close_sphere_768/{rows[data+idx][0][:-4]}_txt2img_interpolation'
            if not os.path.isdir(save_folder):
                print(save_folder)
                command = run_on_gpu(idx, rows[data+idx])
                p = subprocess.Popen(command, shell=True)
                processes.append(p)
            else:
                print(f"close {rows[data+idx][0][:-4]} already exists")

            if not os.path.isdir(save_folder_1):
                print(save_folder_1)
                command = run_on_gpu(idx, rows[data+idx], use_image=True)
                p = subprocess.Popen(command, shell=True)
                processes.append(p)
            else:
                print(f"far {rows[data+idx][0][:-4]} already exists")
        
        # Wait for all processes to finish
        for p in processes:
            p.wait()

        print("All jobs finished.")


if __name__ == "__main__":
    main()