import numpy as np
import glob
import matplotlib.pyplot as plt
import os
import argparse

parser_obj = argparse.ArgumentParser()

parser_obj.add_argument('--inputs', type=str, default='outputs/measures', help='path to folder containing measures')
parser_obj.add_argument('--outputs_folder', type=str, default='curves/', help='path to store adversarial images')
parser_obj.add_argument('--fig_name', type=str, default='plot', help='name of the saved file')
parser_obj.add_argument('--title', type=str, default='plot', help='title of figure')
parser_obj.add_argument('--upper', type=float, default=2, required=False, help="upper limit of distortion to plot")
parser_obj.add_argument('--lower', type=float, default=0, required=False, help="lower limit of distortion to plot")
parser_obj.add_argument('--range', type=int, default=500, required=False, help="a")
parser_obj = parser_obj.parse_args()
folder = parser_obj.inputs
file_name = parser_obj.fig_name
fig_title = parser_obj.title
range = parser_obj.range

def clean_name(fold_name):
    while fold_name[-1]=='/':
        fold_name = fold_name[:-1]
    return(fold_name[-fold_name[::-1].index('/'):])
file_name= folder#[len('outputs/'):-len('/curves/')]

##
colors = ['b','g','k', 'r', 'c--','c', 'y', 'k--', 'c', 'c--','b', 'g','r', 'c','k', 'y', 'g--', 'c', 'c--']
colors = ['g','b', 'r','m', 'y','k--', 'k', 'k--', 'c', 'c--','b', 'g','r', 'c','k', 'y', 'g--', 'c', 'c--']

labels = None

if True:
    plt.rc('text', usetex=True)
    os.environ["PATH"] += os.pathsep + '/usr/bin/latex'
    pgf_with_rc_fonts = {"pgf.texsystem": "pdflatex"}
    plt.rcParams.update(pgf_with_rc_fonts)
    plt.rc('font', family='serif')
fig, axs = plt.subplots(figsize=(12,6))
secax = axs.twinx()

files = glob.glob('{}/*npy'.format(folder+'/curves/'))
files.sort()
print(files)

print("================================")
print("plotting: ",files)
cpt = 0
top = 0
for file_count, file_id in enumerate(files):
    style = ''
    if '_oth.npy' not in file_id and 'npy' in file_id:
    # if '_oth' not in file_id:
        print(file_id)
        if labels == None:
            label_id = ' '.join(file_id.split('_'))[len(folder+'/curves/'):-4]
        else:
            label_id = labels[cpt]

        # if 'base' in label_id:
        #     label_id = None
        # elif 'att' in label_id:
        #     label_id = None
        # elif 'pgd' in label_id.lower():
        #     label_id = None


        file_array = np.load(file_id)
        file_array_temp = np.zeros(range)
        file_array_temp[:] = file_array[:range]
        file_array = file_array_temp
        data_id  = [np.arange(file_array.shape[0]), file_array]
        if 'accuracy' in label_id:
            data_id[1] /= 100
            curve = secax.plot(data_id[0], data_id[1], colors[cpt], label=label_id, linewidth=1)
            secax.set_ylabel('Accuracy', {'fontsize': 14}, color=colors[cpt])
        else:
            curve = axs.plot(data_id[0], data_id[1], colors[cpt], label=label_id, linewidth=1)
            if data_id[1].max()>top:
                top=data_id[1].max()
        cpt+=1
        print(cpt, colors[cpt])

#axs.set_xticks(np.arange(0, upper_limit+0.1, upper_limit/3))
# axs.set_xticks(np.arange(-lower_limit, upper_limit+0.1, (upper_limit+lower_limit)/3))
# axs.set_yticks(np.arange(0, 10, 1))
# if "lr5_10" in folder:
#     axs.set_ylim(30, 40.1)

#axs.set_ylim(0, 3000.1)
"""ici pour les 1 steps"""
axs.set_ylim(0, top)
# axs.set_ylim(0, 100)
secax.set_ylim(0, 1.0)
secax.set_xlim(0, range)

# fig.legend(shadow=False, loc=(0.62, 0.37), handlelength=1.5, fontsize= 14)
#fig.legend(shadow=True, loc=(0.65, 0.25), handlelength=1.5, fontsize= 14)
fig.legend(shadow=False, loc=(0.7, 0.6), handlelength=1.5, fontsize= 14)
print(file_name)
title = ' '.join(file_name[:-8].split('_'))
axs.set_xlabel("Iter", {'fontsize': 14})
axs.set_ylabel('Losses', {'fontsize': 14}, color='k')
#plt.ylabel("Loss", {'fontsize': 14})
plt.grid( linestyle="dashed")
# axs.set_title(fig_title, fontweight="bold")
fig_name = (clean_name(file_name))
plt.savefig("{}/training_{}.jpeg".format(folder, fig_name),bbox_inches='tight')
