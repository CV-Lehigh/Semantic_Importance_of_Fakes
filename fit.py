import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import zscore
from sklearn.ensemble import IsolationForest
import math
import cv2
import os
import matplotlib.gridspec as gridspec
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import t
from tqdm import tqdm
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline
import pickle


def plot(plots, plot_titles, dimension_x, dimension_y, save_name, title):
    x = 5*len(plots)
    plt.figure(figsize=(7, 5))
    plt.suptitle(title, fontsize=12)


    for idx, plot in enumerate(plots):
        plt.subplot(dimension_y, dimension_x, (idx+1))
        plt.title(f"{plot_titles[idx]}")
        plt.imshow(plot)
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(save_name, bbox_inches='tight', pad_inches=.01)

def compute_grid_dimensions(x):
    if x <= 0:
        return (0, 0)
    
    cols = math.ceil(math.sqrt(x))  # Start with square-like columns
    rows = math.floor(x / cols)  # Compute rows based on columns
    
    if rows * cols < x:  # If we don't fit all images, increase rows
        rows += 1
    
    return rows, cols


def gaussian(x, a, mu, sigma):
    return a * np.exp(-((x - mu)**2) / (2 * sigma**2))

def fourier_series(x, a0, a1, b1, a2, b2):
    return a0 + a1 * np.cos(x) + b1 * np.sin(x) + a2 * np.cos(2*x) + b2 * np.sin(2*x)

def exponential(x, a, b):
    return a * np.exp(b * x)

def logarithmic(x, a, b):
    return a * np.log(x) + b

def power_func(x, a, b):
    return a * np.power(x, b)

def sigmoid(x, L, x0, k):
    return L / (1 + np.exp(-k * (x - x0)))


split_score = []
bounds = []
function = []
sorted_x_bounds = []
cases = []
def plot_new_data(sim_data, poly, function_type):
    # Plot
    colours = ['blue', 'red', 'brown','purple']
    labels = ['sphere','sphere','PAID', 'PAID']
    # save_figure = ['close_sphere', 'close_linear', 'close_paid', 'far_sphere', 'far_linear', 'far_paid', 'all']
    ax = ['ax1', 'ax2', 'ax3', 'ax4', 'ax5', 'ax6', 'ax7', 'ax8']

    fig = plt.figure(figsize=(8, 4))

    gs = gridspec.GridSpec(1, 2, figure=fig)
    order = [[1,0], [0,0], [1,1], [0,1]]

    for split in tqdm(range(4)):
    # try:
        if split == 6:
            continue
            # np_array = np.array(sim_data[0])
            # for s in range(1,6):
            #     np_array = np.concatenate((np_array, np.array(sim_data[s])))
            # ax[split] = fig.add_subplot(gs[2, 0:3])


        else:
            np_array =  np.array(sim_data[split])
            ax[split] = fig.add_subplot(gs[order[split][1]])


        x = np_array[:,0]
        y = np_array[:,1]

        model = IsolationForest(contamination=.06, random_state=42)
        outlier_pred = model.fit_predict(np_array[:, :2])

        # -1 are outliers, 1 are inliers
        inliers = np_array[outlier_pred == 1]

        # Degree of the polynomial
        if function_type == 'poly':
            degree = poly
            initial_coeffs = np.polyfit(inliers[:, 0], inliers[:, 1], degree)
            initial_poly = np.poly1d(initial_coeffs)
            sorted_indices = np.argsort(inliers[:, 0])

            sorted_x = inliers[:, 0][sorted_indices]
            sorted_y = inliers[:, 1][sorted_indices]

            with open(f'image_poly_10_func_{split}.pkl', 'wb') as f:
                pickle.dump(initial_poly, f)


            y_pred = initial_poly(sorted_x)

        elif function_type == 'gaus': 
            initial_guess = [1, 0, 1]  # [a, mu, sigma]
            popt, _ = curve_fit(gaussian, inliers[:, 0], inliers[:, 1], p0=initial_guess)

            sorted_indices = np.argsort(inliers[:, 0])
            sorted_x = inliers[:, 0][sorted_indices]
            sorted_y = inliers[:, 1][sorted_indices]

            y_pred = gaussian(sorted_x, *popt)
        
        elif function_type == 'fourier':
            # Fit the Fourier series
            initial_guess = [0, 0, 0, 0, 0]  # Start with all coefficients at zero
            popt, _ = curve_fit(fourier_series, x, y, p0=initial_guess)

            sorted_indices = np.argsort(inliers[:, 0])
            sorted_x = inliers[:, 0][sorted_indices]
            sorted_y = inliers[:, 1][sorted_indices]

            y_pred = fourier_series(sorted_x, *popt)

        elif function_type == 'exp':
            popt, _ = curve_fit(exponential, x, y, p0=(1, 0.1))

            sorted_indices = np.argsort(inliers[:, 0])
            sorted_x = inliers[:, 0][sorted_indices]
            sorted_y = inliers[:, 1][sorted_indices]

            y_pred = exponential(sorted_x, *popt)
        
        elif function_type == 'log':
            popt, _ = curve_fit(logarithmic, x, y, p0=(1, 1))

            sorted_indices = np.argsort(inliers[:, 0])
            sorted_x = inliers[:, 0][sorted_indices]
            sorted_y = inliers[:, 1][sorted_indices]

            y_pred = logarithmic(sorted_x, *popt)
        
        elif function_type == 'power':
            popt, _ = curve_fit(power_func, x, y, p0=(1, 1))

            sorted_indices = np.argsort(inliers[:, 0])
            sorted_x = inliers[:, 0][sorted_indices]
            sorted_y = inliers[:, 1][sorted_indices]

            y_pred = power_func(sorted_x, *popt)
        
        elif function_type == 'sig':
            popt, _ = curve_fit(sigmoid, x, y, p0=(max(y), np.median(x), 1))

            sorted_indices = np.argsort(inliers[:, 0])
            sorted_x = inliers[:, 0][sorted_indices]
            sorted_y = inliers[:, 1][sorted_indices]

            y_pred = sigmoid(sorted_x, *popt)

        mse = mean_squared_error(sorted_y, y_pred)
        mae = mean_absolute_error(sorted_y, y_pred)


        lower_bounds = []
        upper_bounds = []
        # print(indexes)

        for index in range(len(sorted_x)):
            num_samples_bottom = 200
            num_samples_top = 200
            if index < 200:
                num_samples_bottom = index
            if len(sorted_x) - (index+1) < 200:
                num_samples_top = len(sorted_x) - (index+1)
            subset=sorted_y[(index-num_samples_bottom):(index+num_samples_top)] - y_pred[(index-num_samples_bottom):(index+num_samples_top)]
            std_error = np.std(subset)
            lower_bounds.append((y_pred[index]-std_error))
            upper_bounds.append((y_pred[index]+std_error))


        # First derivative (slope function)
        if function_type == 'poly':
            p_derivative = initial_poly.deriv()
            # Find the roots of the derivative (where the slope is 0)
            # Filter x and derivative values within the parameter range
            x_min_param = 0.2 #Limited samples (results in outlier slopes)
            x_max_param = 0.75 #limitied samples (results in outlier slopes)

            filtered_x_vals = sorted_x[(sorted_x >= x_min_param) & (sorted_x <= x_max_param)]
            y_deriv_vals = p_derivative(filtered_x_vals)
            avg_slope = np.mean(y_deriv_vals)

            min_slope_index = np.argmin(y_deriv_vals)
            
            x_min_slope = sorted_x[min_slope_index]
            tolerance = 5
            candidates_right = np.where(np.abs(y_deriv_vals[min_slope_index:] - avg_slope) < tolerance)[0]
            candidates_left = np.where(np.abs(y_deriv_vals[:min_slope_index] - avg_slope) < tolerance)[0]
            if len(candidates_right) > 0:
                right_idx = candidates_right[0] + min_slope_index # earliest index
            if len(candidates_left) > 0:
                left_idx = candidates_left[len(candidates_left)-1] # earliest index

        
        x_selected = [sorted_x[200], sorted_x[int(len(sorted_x)*.20)], sorted_x[int(len(sorted_x)*.40)],sorted_x[int(len(sorted_x)*.60)],sorted_x[int(len(sorted_x)*.80)], sorted_x[int(len(sorted_x)-1)]]
        y_selected = [y_pred[200], y_pred[int(len(sorted_x)*.20)], y_pred[int(len(sorted_x)*.40)],y_pred[int(len(sorted_x)*.60)], y_pred[int(len(sorted_x)*.80)], y_pred[int(len(sorted_x)-1)]]
        ax[split].plot(sorted_x, y_pred, label=f'{labels[split]} best fit - MAE: {mae:.3f}', color=colours[split])
        ax[split].scatter(x_selected, y_selected, color=colours[split])
        if function_type == 'poly':
            # ax[split].axvline(x=x_min_slope, color = 'black', label=f'Cutoff: {x_min_slope:.3f}', alpha=1.0, linewidth=5)
            ax[split].axvline(x=sorted_x[left_idx], color='gray', label = 'Hard Cases',  linestyle='--')
            ax[split].axvline(x=sorted_x[right_idx], color='gray',  linestyle='--')
            split_score.append(x_min_slope)
            print(left_idx)
            print(right_idx)

        # if split == 6:
        ax[split].fill_between(sorted_x, lower_bounds, upper_bounds, color=colours[split], alpha=0.3, label=r'$\pm  1 \sigma$')
        ax[split].set_xlabel('LPIPS', fontsize = 14)
        ax[split].set_ylim(25, 100)
        ax[split].tick_params(axis='both', labelsize=12)
        # ax[split].set_xlim(0.0, 0.8)
        ax[split].set_ylabel('CLIP-Score', fontsize = 14)
        ax[split].legend(loc='lower left', fontsize = 12)
        ax[split].grid(True)
        bounds.append([lower_bounds, upper_bounds])
        cases.append([sorted_x[left_idx], sorted_x[right_idx]])
        sorted_x_bounds.append(sorted_x)

    fig.suptitle(r'Prompt + Image Interpolation Semantic Change', fontsize = 14)
    plt.tight_layout()
    plt.savefig(f'./{function_type}/image_entire_best_fit_{poly}_no_avg_0.png')
    plt.cla()
    plt.clf()



sim_data = np.load('image_sim_data_final.npy', allow_pickle=True)
# for poly in tqdm(range(1,11)):
os.makedirs(f'poly', exist_ok=True)
plot_new_data(sim_data, 10, 'poly')
np.save('image_split_score.npy', np.array(split_score, dtype=object))
np.save('image_bounds.npy', np.array(bounds, dtype=object))
np.save('image_x_bounds.npy',np.array(sorted_x_bounds, dtype=object))
np.save('image_hard_case.npy', np.array(cases, dtype=object))

# plot_new_data(sim_data,0, 'gaus')
# plot_new_data(sim_data,0, 'fourier')
# plot_new_data(sim_data,0, 'exp')
# plot_new_data(sim_data,0, 'log')
# plot_new_data(sim_data,0, 'power')
# plot_new_data(sim_data,0, 'sig')