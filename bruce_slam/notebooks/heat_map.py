import sys
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# parse the arguments, do we want to visulize and do we want to run a quant study
scene = sys.argv[1]

# load the data file
file = open("reports/"+scene+"/coverage.pickle",'rb')
arr = pickle.load(file)

# blank heat map
heat_map = np.zeros((3,15))
    
# loop over the different cloud types
for i,cloud_type in enumerate(["fusion","infer","submap"]):
    
    # loop over the rotations
    for rotation in range(30,120,30):

        # start the counters
        step_fusion = 0
        step_infer = 5
        step_submap = 10
        
        # loop over the translation options
        for translation in range(1,6):

            # parse the data from the dictionary
            vals = arr[(cloud_type,translation,rotation)]
            
            # populate the heat map
            if i == 0:
                heat_map[rotation//30 - 1][step_fusion] = vals[-1]
                step_fusion += 1
            elif i == 1:
                heat_map[rotation//30 - 1][step_infer] = vals[-1]
                step_infer += 1
            elif i == 2:
                heat_map[rotation//30 - 1][step_submap] = vals[-1]
                step_submap += 1


labels = np.array(heat_map)
for i in range(len(labels)):
    for j in range(len(labels[0])):
        labels[i][j] = np.round(labels[i][j] / 10000,2)

sns.set_theme(style="white")

# Generate a large random dataset
d = pd.DataFrame(heat_map)

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(16,5))

# generate a heatmap
#cmap = sns.color_palette("magma", as_cmap=True, n_colors=20)
cmap = sns.color_palette("plasma", as_cmap=True, n_colors=20)

# Draw the heatmap with the mask and correct aspect ratio
sns.set(font_scale=1.5)
sns.heatmap(d, cmap=cmap,square=True, annot = labels, linewidths=.5, cbar_kws={"shrink": 0.5, 'label': 'Voxel Count'})

# add some ticks
plt.xticks([0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5,10.5,11.5,12.5,13.5,14.5],
           ["1","2","3","4","5","1","2","3","4","5","1","2","3","4","5"],fontsize=20)
plt.yticks([0.5,1.5,2.5],["30","60","90"],fontsize=20,rotation=90)

# add the method labels
secax = ax.secondary_xaxis('top')
secax.set_xticks([2.5,7.5,12.5])
secax.set_xticklabels(["Fusion","Inference","Submap"],fontsize=30)

# plot the dividing lines between the methods
for i in range(5,15,5):
    plt.plot([i,i],[0,3],linewidth=8,linestyle="-",color="white")
    
# add the axis labels and log
plt.xlabel("Distance Between Keyframes (meters)",fontsize = 20)
plt.ylabel("Rotation Between \n Keyframes (degrees)",fontsize = 20)
plt.savefig("reports/"+scene+"/heatmap.png")
plt.show()