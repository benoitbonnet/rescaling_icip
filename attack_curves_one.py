import numpy as np
import glob
import matplotlib.pyplot as plt
import os
import argparse

parser_obj = argparse.ArgumentParser()

parser_obj.add_argument('--inputs', type=str, default='outputs/measures', help='path to folder containing measures')
parser_obj.add_argument('--outputs_folder', type=str, default='curves/', help='path to store adversarial images')
parser_obj.add_argument('--fig_name', type=str, default='plot', help='name of the saved file')
parser_obj.add_argument('--upper', type=float, default=0.9, required=False, help="upper limit of distortion to plot")
parser_obj.add_argument('--lower', type=float, default=0, required=False, help="lower limit of distortion to plot")
parser_obj = parser_obj.parse_args()
folder = parser_obj.inputs
file_name = parser_obj.fig_name
upper_limit = parser_obj.upper
lower_limit = parser_obj.lower

def make_data(np_array, up_lim):
    res_array = np.zeros((2, np_array.shape[0]+1))
    np_array.sort()
    cpt = 1
    already_found = True
    for j in range(np_array.shape[0]):
        if np_array[j]==0:
            #res_array[1,j] = ((np_array==0).sum()+1)/10
            res_array[1,j] = (np_array==0).sum()/10
            res_array[0,j] = np_array[j]
        else:
            if already_found == False:
                first_value = np_array[j]
                res_array[0,:j] = first_value
            already_found = True
            res_array[1,j] = 100*cpt/np_array.shape[0]
            res_array[0,j] = np_array[j]
        cpt+=1
    res_array[1,-1] = res_array[1,-2]
    res_array[0,-1] = up_lim
    #print(res_array[:,:10])
    #print(np_array[:10])
    #exit()
    return(res_array)


colors = ['b', 'b--','r', 'r--', 'g', 'g--','m', 'm--', 'k', 'r--', 'k--', 'g--','r--', 'k--','g--', 'c','k', 'y', 'g--', 'c', 'c--']
colors = ['b','g', 'r', 'k', 'm','b--','g--', 'r--', 'k--', 'm--', 'y','g--', 'c','k', 'y', 'g--', 'c', 'c--']
#colors = ['b','g', 'r', 'orange','m', 'k', 'orange', 'g', 'y','r--', 'k--','g--', 'c','k', 'y', 'g--', 'c', 'c--']
labels = None
fig_name = '{}/{}'.format(parser_obj.outputs_folder, file_name)
print('saving plot as {}'.format(fig_name))

if True:
    plt.rc('text', usetex=True)
    os.environ["PATH"] += os.pathsep + '/usr/bin/latex'
    pgf_with_rc_fonts = {"pgf.texsystem": "pdflatex"}
    plt.rcParams.update(pgf_with_rc_fonts)
    plt.rc('font', family='serif')
fig, axs = plt.subplots(figsize=(12,8))

axs.set_xlim([0,upper_limit])
axs.set_xlim([-lower_limit,upper_limit])

files = glob.glob('{}/*npy'.format(folder+'/measures/'))
files.sort()
# print("================================")
print("avant sort: ",files)
#pgd
#files[0],files[3] = files[3],files[0]
#epsilon
#files[0],files[3] = files[3],files[0]
#files[0],files[1],files[2],files[3],files[4],files[5] = files[5],files[0],files[4],files[3],files[1],files[2]
#optim
# files[0],files[3] = files[3],files[0]
for cpt, file_id in enumerate(files):
    style = ''

    if labels == None:
        if 'nat' not in file_id:
            label_id = ' '.join(file_id.split('_'))[len(folder+'/measures/'):-4]
        else:
            label_id = 'Natural'
    else:
        label_id = labels[cpt]
    # elif 'att' in label_id:
    #     label_id = None
    # elif 'pgd' in label_id.lower():
    #     label_id = None


    file_array = np.load(file_id)
    print(label_id, " mean: ", (file_array[file_array<=4]).sum()/(1000-(file_array==0).sum()-(file_array==1e6).sum()), ", failed: ", (file_array>=4).sum(), ", already adv:", (file_array==0).sum())
    data_id  = make_data(file_array, upper_limit)
    #curve = axs.plot(data_id[0,:], data_id[1,:], colors[cpt], label=label_id, linewidth=1)
    if cpt<=2:
        curve = axs.plot(data_id[0,:], data_id[1,:], colors[cpt], label=label_id, linewidth=1)
    else:
        curve = axs.plot(data_id[0,:], data_id[1,:], colors[cpt], label=label_id, linewidth=1)
    cpt+=1

#axs.set_xticks(np.arange(0, upper_limit+0.1, upper_limit/3))
axs.set_xticks(np.arange(-lower_limit, upper_limit+0.1, (upper_limit+lower_limit)/3))
axs.set_yticks(np.arange(0, 100+0.1, 20))
"""Models"""
#fig.legend(shadow=True, loc=(0.45, 0.2), handlelength=1.5, fontsize= 14)
"""steps"""
#fig.legend(shadow=True, loc=(0.65, 0.18), handlelength=1.5, fontsize= 14)
"""DEGREES"""  """QFS"""
#fig.legend(shadow=True, loc=(0.6, 0.3), handlelength=1.5, fontsize= 14)
fig.legend(shadow=False, loc=(0.6, 0.25), handlelength=1.5, fontsize= 20)
print(file_name)
title = ' '.join(file_name[:-8].split('_'))
print(title)
plt.xlabel("Distortion", {'fontsize': 14})
plt.ylabel("success rate ($\%$)", {'fontsize': 20})
plt.grid( linestyle="dashed")
plt.savefig(parser_obj.inputs+"robustesse.jpeg", bbox_inches='tight')
