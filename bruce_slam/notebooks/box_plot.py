import sys
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# parse the arguments, do we want to visulize and do we want to run a quant study
scene = sys.argv[1]

# loop over the data and make it pandas data frames 
# the seaborn box plotter only takes pandas data <eye roll>
data_frames = []
for i, cloud_type in enumerate(["fusion","infer","submap"]):

    # load up the data
    file = open("csv/"+scene+"/"+cloud_type+"_distance_metrics.pickle",'rb')
    table = pickle.load(file)

    # loop and push data into pandas, append to list
    for key in table.keys():
        kf_dist, kf_rot = key
        data_frames.append(
            pd.DataFrame(table[(kf_dist,kf_rot)], 
            columns=[str(kf_dist)+ "," + str(kf_rot)]).assign(Trial=i)
        )

# combine all the dataframes
cdf = pd.concat(data_frames)                              
mdf = pd.melt(cdf, id_vars=['Trial'], var_name=['Keyframe Spacing'])  

# plot the data as a box plot
sns.set(style="dark", palette="pastel", color_codes=True)
plt.figure(figsize=(20,20))
ax = sns.boxplot(x="Trial", y="value", hue="Keyframe Spacing", data=mdf,showfliers=False)
plt.xticks([0,1,2], ["Fusion Clouds", "Inference Method", "Submapping"],fontsize=40)
plt.xlabel("")
plt.ylabel("Absolute Error, Meters",fontsize=40)
plt.xlabel("")
plt.legend(title='Keyframe Spacing', fontsize = 16, loc='center left', bbox_to_anchor=(1, 0.5))
plt.savefig("csv/"+scene+".png")
plt.show()