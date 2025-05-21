import interpolate  # Runs script.py
import numpy as np
import runpy
import sys
import subprocess
from tqdm import tqdm
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed


rows = np.load('./captions_alike.npy')

def assemble_args(data, use_image=False):
    if use_image:
        return  [
        "--prompt_in", f"\"{remove_apostrophes_and_quotes(data[1])}\"",
        "--prompt_out", f"\"{remove_apostrophes_and_quotes(data[3])}\"",
        "--latent_in", f"\"{remove_apostrophes_and_quotes(data[1])}\"",
        "--latent_out", f"\"{remove_apostrophes_and_quotes(data[3])}\"",
        "--idx", data[0][:-4],
        ]

    return  [
        "--prompt_in", f"\"{remove_apostrophes_and_quotes(data[1])}\"",
        "--prompt_out", f"\"{remove_apostrophes_and_quotes(data[3])}\"",
        "--idx", data[0][:-4],
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

def run_task(command):
    subprocess.Popen(command, shell=True)


# for data in tqdm(range(0, len(rows), 4)):
    # Task definition
def run_on_gpu(gpu_id, data, use_image=False):
    script = "YYOUR DATA PATH/attention-interpolation-diffusion/interpolate.py"

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
        for idx in range(4):
            # Run the task for each GPU
            save_folder = f'YOUR DATA PATH/prompt_close_PAID_0/{str(rows[data+idx][0][:-4])}'
            image_save_folder = f'YOUR DATA PATH/image_prompt_close_PAID/{str(rows[data+idx][0][:-4])}'

            if not os.path.isdir(save_folder):
                # print('', end='')
                command = run_on_gpu(idx, rows[data+idx])
                p = subprocess.Popen(command, shell=True)
                processes.append(p)
            else:
                print(f"{rows[data+idx][0][:-4]} already exists")

            if not os.path.isdir(image_save_folder):
                # print('', end='')
                command = run_on_gpu(idx, rows[data+idx])
                p = subprocess.Popen(command, shell=True)
                processes.append(p)
            else:
                print(f"{rows[data+idx][0][:-4]} already exists")
        # Wait for all processes to finish
        for p in processes:
            p.wait()

        print("All jobs finished.")

if __name__ == "__main__":
    main()