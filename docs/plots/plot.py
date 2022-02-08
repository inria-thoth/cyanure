import numpy as np
import os.path 
import matplotlib.pyplot as plt

encoding = { 1: "ista", 2: 'fista', 3: 'catalyst-ista', 4: "qning-ista", 5: 'svrg', 6: 'catalyst-svrg', 7: "qning-svrg", 8: 'acc-svrg', 
        9: "miso", 10: "catalyst-miso", 11: "qning-miso", 12: 'lbfgs', 13: "liblinear", 14: "saga", 15: "newton-cg", 16: "liblinear-dual"}
encoding_legend = { 1: "ista", 2: 'fista', 3: 'catalyst-ista', 4: "qning-ista", 5: 'svrg', 6: 'catalyst-svrg', 7: "qning-svrg", 8: 'acc-svrg', 
            9: "miso", 10: "catalyst-miso", 11: "qning-miso", 12: 'lbfgs', 13: "liblinear", 14: "sklearn-saga", 15: "newton-cg", 16: "liblinear-dual"}
dataset_names = ['covtype', 'alpha', 'real-sim', 'epsilon', 'ocr', 'rcv1', 'webspam', 'kddb', 'criteo', 'ckn_mnist', 'svhn']

line_colors = { 2: 'brown', 4: 'blue', 5: 'firebrick', 6: 'firebrick', 7: 'firebrick', 8: 'black', 10: 'royalblue',
        11: "royalblue", 12: "green", 13: 'purple', 14: "orange", 15: "darkgray", 9: "darkviolet", 16: "red"}

line_styles = { 2: '-', 4: '--', 5: '-', 6: '--', 7: ':', 8: '-', 10: '--',  11: ":", 12: "None", 13: 'None', 14: "None",
        15: 'None', 9:'--', 16: 'None'}
marker_styles = { 12: 'o', 13: "*", 14: "^", 15: "X", 16: "X"}
solvers=[5,6,7,8,10,11,12,13,14,15,16]

penalty='l2'
logpath='./logs_arsenic/'
min_trim=1e-7
for dataset in dataset_names:
    list_solvers=[]
    #print(dataset)
    min_eval=100
    max_dual=-100
    for solver in solvers:
        if solver >= 12:
            print(encoding[solver])
            namelog=logpath+dataset+"_"+encoding[solver]+"_" + penalty + "_optimlambda_seed0_tol0.001.npy"
            if os.path.isfile(namelog):
                optim=np.load(namelog)
                min_eval=min(min_eval,np.min(optim[1,]))
                max_dual=max(max_dual,np.max(optim[2,]))
            namelog=logpath+dataset+"_"+encoding[solver]+"_" + penalty + "_optimlambda_seed0_tol0.0001.npy"
            if os.path.isfile(namelog):
                optim=np.load(namelog)
                min_eval=min(min_eval,np.min(optim[1,]))
                max_dual=max(max_dual,np.max(optim[2,]))
        else:
            namelog=logpath+dataset+"_"+encoding[solver]+"_" + penalty + "_optimlambda_seed0.npy"
            if os.path.isfile(namelog):
                optim=np.load(namelog)
                print(optim)
                min_eval=min(min_eval,np.min(optim[1,]))
                max_dual=max(max_dual,np.max(optim[2,]))
    print(min_eval)
    print(max_dual)
    for solver in solvers:
        if solver >= 12:
            optim=np.empty(0)
            for tol in [0.1,0.01,0.001,0.0001]:
                namelog=logpath+dataset+"_"+encoding[solver]+"_" + penalty + "_optimlambda_seed0_tol" + str(tol) + ".npy"
                print(namelog)
                if os.path.isfile(namelog) and (solver != 16 or tol==0.1):
                #if os.path.isfile(namelog):
                    if optim.size == 0:
                        optim=np.load(namelog)
                    else:
                        optim2=np.load(namelog)
                        optim=np.concatenate((optim,optim2),axis=1)
            if optim.size > 0:
                print('tete')
                plt.figure(1)
                plt.plot(optim[5,],np.maximum((optim[1,]-max_dual)/min_eval,1e-9),c=line_colors[solver],ls=line_styles[solver],marker=marker_styles[solver],scaley=False)
                list_solvers.append(solver)
        else:
            namelog=logpath+dataset+"_"+encoding[solver]+"_" + penalty + "_optimlambda_seed0.npy"
            if os.path.isfile(namelog):
                optim=np.load(namelog)
                plt.figure(1)
                plt.plot(optim[5,],np.maximum((optim[1,]-max_dual)/min_eval,1e-9),c=line_colors[solver],ls=line_styles[solver],scaley=False)
                list_solvers.append(solver)
            #namelog=logpath+dataset+"_"+encoding[solver]+"_" + penalty + "_optimlambda_seed0.npy"
            #if os.path.isfile(namelog):
            #    optim=np.load(namelog)
            #    plt.figure(1)
            #    plt.plot(optim[5,],np.maximum((optim[1,]-max_dual)/min_eval,1e-9),c='k',ls=line_styles[solver],scaley=False)
            #    list_solvers.append(solver)


    plt.legend(list(encoding_legend[solver] for solver in list_solvers))
    plt.ylabel('Relative optimality gap')
    plt.xlabel('Time in seconds - Dataset '+dataset)
    plt.yscale('log')
    plt.xscale('log')
    axes = plt.gca()
    #axes.set_ylim(ymin=1e-9-1e-10)
    #axes.set_ylim(ymin=1e-9-1e-10)
    xmin, xmax=axes.get_xlim()
    plt.hlines(1e-3, xmin, xmax, colors='k', linestyles='dotted')
    plt.savefig('figs/' + dataset + '_logistic_' + penalty + '.eps', bbox_inches='tight',transparent=True)
    plt.savefig('figs/' + dataset + '_logistic_' + penalty + '.png',dpi=100, bbox_inches='tight',transparent=True)
    plt.show()



