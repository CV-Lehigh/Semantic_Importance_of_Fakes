import os
import numpy as np
import pickle
import random
from collections import defaultdict


def make_zipped_triplets_and_doubles(arr: np.ndarray):
    """
    arr: N×3 object array of [anchor, manip, label]
    Returns a list where each element is either
      [anchor, [neg,0], [pos,1]]   # one-to-one zipped
    or [anchor, [manip,label]]     # fallback doubles
    """
    # group manipulations per anchor
    groups = defaultdict(list)
    for anchor, manip, lab in arr:
        groups[anchor].append((manip, int(lab)))

    out = []
    for anchor, lst in groups.items():
        negs  = [m for m,l in lst if l == 0]
        poss  = [m for m,l in lst if l == 1]

        # if we have at least one of each, zip them in order
        if negs and poss:
            for n, p in zip(negs, poss):
                out.append([anchor, [n, 0], [p, 1]])
            # any leftover (if lengths differ) become doubles
            extra = lst[len(negs):] if len(negs)>len(poss) else lst[len(poss):]
            for m,l in extra:
                out.append([anchor, [m, l]])
        else:
            # no valid pair → all as doubles
            for m,l in lst:
                out.append([anchor, [m, l]])
    return out


easy_items = []
hard_items = []

#generates dataloader with all data
sets = ['caption_', 'image_']
selection = 'all'


##generates dataloader with prompt only data
# sets = ['caption_']
# selection = 'caption'



##generates dataloader with prompt + image only data
# sets = ['image_']
# selection = 'image'



for li in sets:
    add_in = li
    non_matching_scores = np.load(f'{add_in}/{add_in}split_score.npy',allow_pickle=True)
    cases = np.load(f'{add_in}/{add_in}hard_case.npy', allow_pickle=True)
    file_paths = np.load(f'{add_in}/{add_in}file_paths_final.npy',allow_pickle=True)
    sim_data = np.load(f'{add_in}/{add_in}sim_data_final.npy',allow_pickle=True)
    bounds = np.load(f'{add_in}/{add_in}bounds.npy', allow_pickle=True)
    x_bounds = np.load(f'{add_in}/{add_in}x_bounds.npy', allow_pickle=True)


    # decide if data is a good canidate
    # construct a list (origianl image, img_path, matching (0 or 1))
    #### 0 == matching
    #### 1 == non-matching

    def percentage_through_range(x, y, z):
        return ((z - x) / (y - x))

    i_value = 0
    ##split based on interpoltion method
    for split in range(len(non_matching_scores)):
        with open(f'{add_in}/{add_in}poly_10_func_{split}.pkl', 'rb') as f:
            best_fit = pickle.load(f)
        cut_off = non_matching_scores[split]

        
        for index in range(len(sim_data[split])):
            file_name = file_paths[split][index]
            if 'close' not in file_name[0]:
                continue
            true_y = sim_data[split][index][1]
            x_value = sim_data[split][index][0]
            idx = np.abs(x_bounds[split] - sim_data[split][index][0]).argmin()
            left_bound = cases[split][0].item()
            right_bound = cases[split][1].item()
            if true_y < bounds[split][1][idx] and true_y > bounds[split][0][idx]:
                if x_value >= left_bound and x_value <= right_bound:
                    x_cutoff = percentage_through_range(left_bound, right_bound, x_value)
                    chance = random.random()
                    if chance >= x_cutoff:
                        hard_items.append([file_paths[split][index][0], file_paths[split][index][1], 0])
                    else:
                        hard_items.append([file_paths[split][index][0], file_paths[split][index][1], 1])


                elif x_value > right_bound:
                    easy_items.append([file_paths[split][index][0], file_paths[split][index][1], 1])

                elif x_value < left_bound:
                    easy_items.append([file_paths[split][index][0], file_paths[split][index][1], 0])

new_easy = make_zipped_triplets_and_doubles(easy_items)
new_hard = make_zipped_triplets_and_doubles(hard_items)

np.save(f'./data_selection/1std_{selection}_hard_cases_cnt_random_trip_close_only_data.npy', np.array(new_hard, dtype=object))
np.save(f'./data_selection/1std_{selection}_easy_cases_cnt_random_trip_close_only_data.npy', np.array(new_easy, dtype=object))

combined = new_easy + new_hard
unique_anchors = set(item[0] for item in combined)
print(f"Number of unique first elements: {len(unique_anchors)}")

