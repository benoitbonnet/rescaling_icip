import numpy as np
import glob
import matplotlib.pyplot as plt
import os


colors = ['g','b', 'r','m', 'y','k--', 'k', 'k--', 'c', 'c--','b', 'g','r', 'c','k', 'y', 'g--', 'c', 'c--']
labels = None


if True:
    plt.rc('text', usetex=True)
    os.environ["PATH"] += os.pathsep + '/usr/bin/latex'
    pgf_with_rc_fonts = {"pgf.texsystem": "pdflatex"}
    plt.rcParams.update(pgf_with_rc_fonts)
    plt.rc('font', family='serif')

for net in ['resnet', 'effnet', 'resnet_rob']:
    for eps in ['05','1','4', '100']:
        fig, axs = plt.subplots(figsize=(12,6))

        folder = 'outputs/layer_sensibility/{}/measures/eps{}/'.format(net,eps)
        files = glob.glob('{}/*npy'.format(folder))
        files.sort()

        print("plotting: ",eps)
        ref = np.load(folder+'clean.npy')
        # ref = ref.mean(1)

        for cpt, file_id in enumerate(files):
            style = ''
            # if 'adv' in file_id:
            if True:


                if labels == None:
                    label_id = ' '.join(file_id.split('_'))[len(folder):-4]
                else:
                    label_id = labels[cpt]

                if 'base' in label_id:
                    label_id = None
                # elif 'att' in label_id:
                #     label_id = None
                elif 'pgd' in label_id.lower():
                    label_id = None


                file_array = np.load(file_id).mean(1)
                data_id  = [np.arange(file_array.shape[0]), file_array]
                #curve = axs.plot(data_id[0,:], data_id[1,:], colors[cpt], label=label_id, linewidth=1)
                if label_id is not 'clean':
                    curve = axs.plot(data_id[0], data_id[1], colors[cpt], label=label_id, linewidth=1)
                #xvalues = curve[0].get_xdata()
                #yvalues = curve[0].get_ydata()
                cpt+=1


        #axs.set_xticks(np.arange(0, upper_limit+0.1, upper_limit/3))
        # axs.set_xticks(np.arange(-lower_limit, upper_limit+0.1, (upper_limit+lower_limit)/3))
        # axs.set_yticks(np.arange(0, 100+0.1, 50))
        axs.set_ylim(0, 1.1)


        fig.legend(shadow=True, loc=(0.2, 0.55), handlelength=1.5, fontsize= 14)

        # title = ' '.join(file_name[:-8].split('_'))
        plt.xlabel("Loss", {'fontsize': 14})
        plt.ylabel("Epoch", {'fontsize': 14})
        plt.grid( linestyle="dashed")
        # axs.set_title(fig_title, fontweight="bold")
        plt.savefig("outputs/layer_sensibility/{}/measures/sensi_curve_eps{}.jpeg".format(net, eps),bbox_inches='tight')
