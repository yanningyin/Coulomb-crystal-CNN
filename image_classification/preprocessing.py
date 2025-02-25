#!/usr/bin/env python3
"""This module contains functions for performing some useful tasks."""

import os
import glob
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import math
from PIL import Image
from collections import Counter

# image_path = os.path.join(os.path.dirname(__file__), "images_0508half")
# images = glob.glob(os.path.join(image_path, 'numIons*_target_temp*_avgT*mK.tif'))
images = []

# define labels
labels_num_ions = []
labels_target_temp = []
labels_avg_temp = []


def load_data(range_num_ions: range = None, range_target_temp: range = None, range_avg_temp: range = None,
              dir_name='images', check_image: bool = True):
    global images, labels_num_ions, labels_target_temp, labels_avg_temp

    # find the images
    if type(dir_name) == str:
        image_path = os.path.join(os.path.dirname(__file__), dir_name)
        images_all = glob.glob(os.path.join(image_path, 'numIons*_*_avgT*mK*.tif'))
    elif type(dir_name) == list:
        images_all = []
        for d in dir_name:
            image_path = os.path.join(os.path.dirname(__file__), d)
            images_all += glob.glob(os.path.join(image_path, 'numIons*_*_avgT*mK*.tif'))
    else:
        print("Error: dir_name should be either a string or a list")
        return

    print(f"found in total {len(images_all)} images in ./{dir_name}/")
    
    # Loop over the file names and extract the information
    for image in images_all:
        image_basename = os.path.basename(image)
        parts = image_basename.split("_")
        num_ion = int(parts[1])
        target_temp = int(parts[3])
        avg_temp = int(parts[5].split('mK')[0])
        # avg_temp = int(parts[5].replace("mK.tif", ""))
        if check_image and target_temp != avg_temp:
            print(f"Image ignored (target temperature not reached): {image_basename}")
        elif check_image and not _is_temp_stable(image, num_ion, target_temp, avg_temp):
            print(f"Image ignored (temperature not stable enough): {image_basename}")
        else:
            # Another way of extracting values from file names is to use regular expression
            images.append(image)
            labels_num_ions.append(num_ion)
            labels_target_temp.append(target_temp)
            labels_avg_temp.append(avg_temp)

    if range_num_ions or range_avg_temp or range_target_temp:
        _filter_data(range_num_ions, range_target_temp, range_avg_temp)

    print(f"loaded {len(images)} images from ./{dir_name}/")

    return images, labels_num_ions, labels_target_temp, labels_avg_temp


# plot the histogram of labels to check its distribution
def view_labels_distribution():
    labels_num_ions_counter = Counter(labels_num_ions)
    labels_target_temp_counter = Counter(labels_target_temp)
    labels_avg_temp_counter = Counter(labels_avg_temp)
    # print(labels_num_ions_counter)
    # print(labels_avg_temp_counter)
    # print(labels_avg_temp_counter)

    fig, axs = plt.subplots(1, 3, figsize=(16, 4))

    # Plot the histograms on each subplot
    # axs[0].hist(labels_num_ions, bins=range(min(labels_num_ions), max(labels_num_ions)), alpha=0.5)
    axs[0].bar(list(labels_num_ions_counter.keys()), labels_num_ions_counter.values())
    axs[0].set_title('labels_num_ions')

    # axs[1].hist(labels_target_temp, bins=range(min(labels_target_temp), max(labels_target_temp)), alpha=0.5)
    axs[1].bar(list(labels_target_temp_counter.keys()), labels_target_temp_counter.values())
    axs[1].set_title('labels_target_temp')

    # axs[2].hist(labels_avg_temp, bins=range(min(labels_avg_temp), max(labels_avg_temp)), alpha=0.5)
    axs[2].bar(list(labels_avg_temp_counter.keys()), labels_avg_temp_counter.values())
    axs[2].set_title('labels_avg_temp')

    plt.show()


# view the images of a certain class
def view_images_for_class(num_ions: int = 0, avg_temp: int = 0, target_temp: int = -1):
    if not num_ions and not avg_temp and target_temp == -1:
        print("error: at least one argument is required.")
        return

    subplot_titles = []
    if not avg_temp and target_temp == -1:
        images_of_class = [images[i] for i, n in enumerate(labels_num_ions) if n == num_ions]
        avg_tmp_of_class = [labels_avg_temp[i] for i, n in enumerate(labels_num_ions) if n == num_ions]
        target_temp_of_class = [labels_target_temp[i] for i, n in enumerate(labels_num_ions) if n == num_ions] 
        images_of_class, avg_tmp_of_class = zip(*sorted(list(zip(images_of_class, avg_tmp_of_class)), key=lambda x: x[1]))
        subplot_titles = [f"T: {t} mK" for t, h in zip(avg_tmp_of_class, target_temp_of_class)]
        fig_title = f"Images for class: num_ions = {num_ions}"
    elif not num_ions and target_temp == -1:
        images_of_class = [images[i] for i, t in enumerate(labels_avg_temp) if t == avg_temp]
        target_temp_of_class = [labels_target_temp[i] for i, t in enumerate(labels_avg_temp) if t == avg_temp]
        num_ions_of_class = [labels_num_ions[i] for i, t in enumerate(labels_avg_temp) if t == avg_temp]
        images_of_class, num_ions_of_class = zip(*sorted(list(zip(images_of_class, num_ions_of_class)), key=lambda x: x[1]))
        subplot_titles = [f"N: {n}" for n, h in zip(num_ions_of_class, target_temp_of_class)]
        fig_title = f"Images for class: avg_temp = {avg_temp}"
    elif not num_ions and not avg_temp:
        images_of_class = [images[i] for i, h in enumerate(labels_target_temp) if h == target_temp]
        avg_temp_of_class = [labels_avg_temp[i] for i, h in enumerate(labels_target_temp) if h == target_temp]
        num_ions_of_class = [labels_num_ions[i] for i, h in enumerate(labels_target_temp) if h == target_temp]
        subplot_titles = [f"N: {n}, T: {t}" for n, t in zip(num_ions_of_class, avg_temp_of_class)]
        fig_title = f"Images for class: target_temp = {target_temp}"
    elif target_temp == -1:
        images_of_class = [images[i] for i, (n, t) in enumerate(zip(labels_num_ions, labels_avg_temp)) if
                           n == num_ions and t == avg_temp]
        target_temp_of_class = [labels_target_temp[i] for i, (n, t) in enumerate(zip(labels_num_ions, labels_avg_temp))
                                if
                                n == num_ions and t == avg_temp]
        fig_title = f"Images for class: num_ions = {num_ions}, avg_temp = {avg_temp}"
        subplot_titles = [f"H:{h}" for h in target_temp_of_class]
    elif not num_ions:
        images_of_class = [images[i] for i, (h, t) in enumerate(zip(labels_target_temp, labels_avg_temp)) if
                           h == target_temp and t == avg_temp]
        num_ions_of_class = [labels_num_ions[i] for i, (h, t) in enumerate(zip(labels_target_temp, labels_avg_temp)) if
                             h == target_temp and t == avg_temp]
        fig_title = f"Images for class: target_temp = {target_temp}, avg_temp = {avg_temp}"
        subplot_titles = [f"N:{n}" for n in num_ions_of_class]
    elif not avg_temp:
        images_of_class = [images[i] for i, (h, n) in enumerate(zip(labels_target_temp, labels_num_ions)) if
                           h == target_temp and n == num_ions]
        avg_temp_of_class = [labels_avg_temp[i] for i, (h, n) in enumerate(zip(labels_target_temp, labels_num_ions)) if
                             h == target_temp and n == num_ions]
        fig_title = f"Images for class: target_temp = {target_temp}, num_ions = {num_ions}"
        subplot_titles = [f"T:{t}" for t in avg_temp_of_class]
    else:
        images_of_class = [images[i] for i, (n, h, t) in
                           enumerate(zip(labels_num_ions, labels_target_temp, labels_avg_temp)) if
                           n == num_ions and h == target_temp and t == avg_temp]
        fig_title = f"Images for class: num_ions = {num_ions}, target_temp = {target_temp}, num_ions = {num_ions}"
        subplot_titles = [" " for i in images_of_class]

    if not images_of_class:
        print("no images found for the specified class")
        print("class for num_ions: ", set(labels_num_ions))
        print("class for avg_temp:", set(labels_avg_temp))
        print("class for target_temp:", set(labels_target_temp))
        return

    # determine the number of rows and columns for the subplots
    num_images = len(images_of_class)
    max_num_subplots = 48
    if num_images <= max_num_subplots:
        print(f"found {len(images_of_class)} images for the specified class")
    else:
        print(f"found {len(images_of_class)} images, only plot the first {max_num_subplots} ones")
        num_images = min([max_num_subplots, num_images])

    rows = int(num_images ** 0.5)
    cols = int(num_images / rows) + int(num_images % rows > 0)

   # create a new figure and set the size
    # fig = plt.figure(figsize=(10, 10))
    fig = plt.figure()
    # loop over the images and add them to the subplots
    for i in range(1, num_images + 1):
        img = Image.open(images_of_class[i - 1])
        ax = fig.add_subplot(rows, cols, i)
        ax.imshow(img)
        ax.set_title(subplot_titles[i - 1], fontdict={'fontsize': 10})
        ax.set_xticks([])
        ax.set_yticks([])

    fig.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.92, wspace=0.1, hspace=0.2)
    # fig.tight_layout()
    # fig.suptitle(fig_title, fontsize=10)
    plt.savefig(f'{fig_title}.pdf', format='pdf')
    print(f'figure saved as {fig_title}.pdf')
    plt.show()


def _is_temp_stable(image, num_ion, target_temp, avg_temp):
    image_path = os.path.dirname(image)
    info_file = glob.glob(os.path.join(image_path, f"numIons_{num_ion}_targetT_{target_temp}.process.info"))
    if not info_file or len(info_file) > 1:
        print("Warning: zero or more than one info file found, proceed without checking info")
        return True

    start_marker = '---'
    found_marker = False
    temp = []
    with open(info_file[0], 'r') as f:
        for line in f:
            if start_marker in line:
                found_marker = True
                continue
            if found_marker:
                columns = line.split('\t')
                if len(columns) > 1:
                    temp.append(float(columns[1]))
    for t in temp:
        if abs(t - avg_temp) > 1:
            return False
    return True

def _filter_data(rN, rH, rT) -> None:
    global images, labels_num_ions, labels_target_temp, labels_avg_temp
    if rN:
        images = [images[i] for i, n in enumerate(labels_num_ions) if n in rN]
        labels_avg_temp = [labels_avg_temp[i] for i, n in enumerate(labels_num_ions) if n in rN]
        labels_target_temp = [labels_target_temp[i] for i, n in enumerate(labels_num_ions) if n in rN]
        labels_num_ions = [labels_num_ions[i] for i, n in enumerate(labels_num_ions) if n in rN]

    if rH:
        images = [images[i] for i, h in enumerate(labels_target_temp) if h in rH]
        labels_avg_temp = [labels_avg_temp[i] for i, h in enumerate(labels_target_temp) if h in rH]
        labels_num_ions = [labels_num_ions[i] for i, h in enumerate(labels_target_temp) if h in rH]
        labels_target_temp = [labels_target_temp[i] for i, h in enumerate(labels_target_temp) if h in rH]

    if rT:
        images = [images[i] for i, t in enumerate(labels_avg_temp) if t in rT]
        labels_num_ions = [labels_num_ions[i] for i, t in enumerate(labels_avg_temp) if t in rT]
        labels_target_temp = [labels_target_temp[i] for i, t in enumerate(labels_avg_temp) if t in rT]
        labels_avg_temp = [labels_avg_temp[i] for i, t in enumerate(labels_avg_temp) if t in rT]


def view_relation_target_temp_temp():
    plt.figure()
    for n in list(set(labels_num_ions)):
        target_temps = [h * h for i, h in enumerate(labels_target_temp) if labels_num_ions[i] == n]
        avg_temps = [t for i, t in enumerate(labels_avg_temp) if labels_num_ions[i] == n]
        plt.scatter(target_temps, avg_temps, label=f'num_ions: {n}')

    plt.xlabel('target_temp')
    plt.ylabel('avg temp (mK)')
    plt.title("Relation avg_temp vs target_temp^2")
    # plt.legend()
    plt.show()


def main():
    # load_data(dir_name="images_0508half", range_avg_temp=range(10, 50))
    # load_data()
    # images_list, labels_num_ions_list, _, _ = load_data(dir_name=["images/images_100_299_5_9/", "images/images_100_299_10_20/"], range_num_ions=range(100, 270, 20), range_avg_temp=range(5, 14, 1), check_image=True)
    images_list, labels_num_ions_list, _, _ = load_data(dir_name=["../images/images_100_299_5_9/", "../images/images_100_299_10_20/"], range_num_ions=range(100, 280, 20), range_avg_temp=range(5, 13, 1), check_image=True)
    # print(images_list[:20])
    # print(labels_num_ions_list[:20])
    view_labels_distribution()
    #view_images_for_class(num_ions=200)
    view_images_for_class(avg_temp=10)
    # view_relation_target_temp_temp()


if __name__ == '__main__':
    main()
