#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 15:07:33 2022

@author: Michael
"""
import matplotlib.animation as animation
from matplotlib import pyplot as plt
from matplotlib.animation import HTMLWriter
import numpy as np

def analyze(infile='Aggregate.txt', Anim=True, AggPlot=True):

    x_lim = [-5, 5]
    y_lim = x_lim
    file = infile
    num_clusters = []
    times = [0]
    
    size = 0
    flag = 0
    with open(file,'r') as f:
        
        f.readline(); 
        vals = f.readline().split()
        r_plot = float(vals[1])/10 * 72 * 72 
        vals = f.readline().split()
        num_clusters.append(int(vals[-1]))
        for line in f:
            if line[0] == 'T':
                break
            elif flag == 0:
                size += 1
            
    if Anim:
        metadata = dict(title='Movie Test', artist='Matplotlib', comment='Movie support')
        writer = HTMLWriter(fps=20, metadata = metadata)  
        fig = plt.figure()
        fig.set_size_inches(5,5)
        #dpi = 100 #default value in matplotlib
        ax = plt.gca()
        ax.set_xlim(x_lim[0], x_lim[1])
        ax.set_ylim(y_lim[0], y_lim[1])
        ax.set_xticks(np.arange(x_lim[0], x_lim[1]+1))
        ax.set_yticks(np.arange(y_lim[0], y_lim[1]+1))
        locs = np.zeros((size,2))
        l = plt.scatter(locs[:,0], locs[:,1], s=r_plot)
        with open(file, 'r') as f:
            with writer.saving(fig, "writer_test.html", 100):
                f.readline()
                f.readline()
                f.readline()
                i = 0
                for line in f:
                    vals = line.split()
                    if i == size:
                      l.set_offsets(locs)
                      writer.grab_frame()
                      i = 0
                    else:
                        locs[i,:] = [float(val) for val in vals[0:2]]
                        i = i + 1
    if AggPlot:
        with open(file, 'r') as f:
            f.readline()
            f.readline()
            f.readline()
            i = 0
            for line in f:
                if line[0] == 'T':
                    vals = line.split()
                    times.append(float(vals[1]))
                    num_clusters.append(int(vals[-1]))
    
        fig2,ax2 = plt.subplots()
        ax2.plot(times, num_clusters)
        ax2.set_xlabel('Simulation Time')
        ax2.set_ylabel('Number of Clusters')
        ax2.set_title('Aggregation Behavior')

analyze(Anim=False)