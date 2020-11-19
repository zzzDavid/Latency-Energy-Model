import math
import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
import shutil
import statistics as st
from matplotlib.pyplot import rcParams
from glob import glob
import re

rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['DejaVu Serif']

latency_dir = './latency'
plot_dir = './plot'

bin_size = 0.0001


def draw_distribution(data, bin_size, file_name):
    """ data should be a list of numbers """
   
    start = int(min(data) / bin_size) * bin_size
    bin_num = int((max(data) - min(data)) / bin_size) + 1
    bins = list()
    for _ in range(bin_num):
        bins.append(list())

    for index, bin in enumerate(bins):
        lower_bound = start + index * bin_size
        upper_bound = lower_bound + bin_size
        for _ in data:
            if not _ >= lower_bound: continue
            if not _ <  upper_bound: continue
            bin.append(_)

    fig, ax = plt.subplots()
    plt.style.use('seaborn-paper')
    fig.set_size_inches(8,5)
    ax.autoscale(True)

    # plot rectangles
    lengths = list()
    for index, bin in enumerate(bins):
        x = start + index * bin_size
        y = 0
        width = bin_size
        height = len(bin)
        lengths.append(height)
        rect = patches.Rectangle((x,y), width, height, linewidth=1, fill=True, facecolor='#87CEBB', alpha=1)
        ax.add_patch(rect)
    
    font_size = 15
    ax.set_xlabel("latency", fontsize=font_size)
    ax.set_ylabel('distribution', fontsize=font_size)
    base_name = os.path.basename(file_name)
    base_name = os.path.splitext(base_name)[0]
    mean = st.mean(data)
    var  = st.variance(data)
    std_var = math.sqrt(var)
    plt.xlim(min(data), max(data))
    plt.ylim(0,max(lengths))
    ax.set_title(f'{base_name} mean={mean:.4f}, var={std_var:.5f}')

    plt.savefig(file_name)
    print('saved figure: ' + file_name)
    


def draw_timeline(data, file_name):
    
    size = 1
    
    fig, ax = plt.subplots()
    plt.style.use('seaborn-paper')
    fig.set_size_inches(8,5)
    ax.autoscale(True)

    # plot rectangles
    lengths = list()
    for index, datum in enumerate(data):
        x = index * size
        y = 0
        width = size
        height = datum
        lengths.append(height)
        rect = patches.Rectangle((x,y), width, height, linewidth=1, fill=True, facecolor='#87CEBB', alpha=1)
        ax.add_patch(rect)
    
    font_size = 15
    ax.set_xlabel("time step", fontsize=font_size)
    ax.set_ylabel('latency', fontsize=font_size)
    base_name = os.path.basename(file_name)
    base_name = os.path.splitext(base_name)[0]
    mean = st.mean(data)
    var  = st.variance(data)
    std_var = math.sqrt(var)
    plt.xlim(0, len(data) + 1)
    plt.ylim(0, max(data))
    ax.set_title(f'{base_name} mean={mean:.4f}, var={std_var:.5f}')

    plt.savefig(file_name)
    print('saved figure: ' + file_name)
 



def parse(latency_file):
    "given a latency result file, return a list of latency numbers"
    data = list()
    with open(latency_file, 'r') as f:
        for line in f.readlines():
            if 'GPU latency' not in line: continue
            # import ipdb; ipdb.set_trace()
            latency_str = line.split('-')[2]
            latency_str = latency_str.replace('GPU latency:','')
            latency_str = latency_str.replace('ms', '')
            latency_str = re.split(f'\s+', latency_str)
            latency_str = [s for s in latency_str if s != '']
            if len(latency_str) == 0: continue
            latency = float( latency_str[0] )
            data.append(latency)
    return data
   
def draw_all_latency_distribution(force=True):
    # prepare plot directory
    if os.path.exists(plot_dir) and force:
        if os.path.exists(plot_dir): shutil.rmtree(plot_dir)
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
 
    for latency_file in glob(os.path.join(latency_dir, '*.txt')):
        base_name = os.path.basename(latency_file)
        base_name = os.path.splitext(base_name)[0]
        plot_file = os.path.join(plot_dir, base_name + '.pdf')
        timeline  = os.path.join(plot_dir, 'timeline' + base_name + '.pdf')
        data = parse(latency_file)
        draw_distribution(data, bin_size, plot_file)
        draw_timeline(data, timeline)

if __name__ == "__main__":
    draw_all_latency_distribution()





