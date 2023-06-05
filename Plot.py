import time

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib as mat
from pylab import cm



def setPlotStyle():
    mat.rcParams['font.size'] = 10
    mat.rcParams['legend.fontsize'] = 15
    mat.rcParams['lines.linewidth'] = 2
    mat.rcParams['lines.color'] = 'r'
    mat.rcParams['axes.grid'] = 1
    mat.rcParams['axes.xmargin'] = 0.1
    mat.rcParams['axes.ymargin'] = 0.1
    mat.rcParams["mathtext.fontset"] = "dejavuserif"  # "cm", "stix", etc.
    mat.rcParams['figure.dpi'] = 300
    mat.rcParams['savefig.dpi'] = 300


def plot_loss(loss_history, lossname):
    # loss_u
    x_len = torch.arange(len(loss_history))

    plt.plot(x_len, torch.log10(torch.tensor(loss_history)), c='blue', label="Training Loss for u")
    # plt.grid()
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    # plt.yticks(torch.tensor([torch.log10(1e-2), torch.log10(1e-1), torch.log10(1e-0)]),
    #            labels=['$10^{-2}$', '$10^{-1}$', '$1$'])
    plt.savefig(time.strftime('./Figure/Training_Loss_u%Y-%m-%d-%H-%M-%S' + lossname), dpi=200)



def plot_2d(XXX, YYY, NO_KI_ELS_mat, uname):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(XXX, YYY, NO_KI_ELS_mat, cmap=cm.coolwarm)

    ax.view_init(20, -135)
    ax.set_zlim3d(0.0, 130.0)
    ax.set_xlabel('S1', fontsize=12)
    ax.set_ylabel('S2', fontsize=12)

    ax.set_xticks(torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]))
    ax.set_xticklabels(['$0$', '$50$', '$100$', '$150$', '$200$', '$250$', '$300$'])
    ax.set_yticks(torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]))
    ax.set_yticklabels(['$0$', '$50$', '$100$', '$150$', '$200$', '$250$', '$300$'])
    ax.set_title('ELS Price'+uname)
    #     fig.savefig('PINN_u_maturity.png')
    print('./Figure/' + uname + time.strftime('%Y-%m-%d-%H-%M-%S'))
    plt.savefig('./Figure/' + uname + time.strftime('%Y-%m-%d-%H-%M-%S'), dpi=200)

def plot_ELS(T, L, N, device, facevalue, NN, uname):
    with torch.autograd.no_grad():
        ONES = torch.ones((N ** 2, 1))
        # ZEROS = torch.zeros((121 ** 2, 1))
        xxx = torch.linspace(0., L, N).reshape(N, 1) / L * L * 1
        yyy = xxx
        XXX, YYY = np.meshgrid(xxx, yyy)
        XXX_flat = XXX.flatten()[:, None]
        YYY_flat = YYY.flatten()[:, None]
        XY = np.concatenate((XXX_flat, YYY_flat), axis=1)
        TXY = np.concatenate((T * ONES, XY), axis=1)
        TXY = torch.FloatTensor(TXY).to(device)

        ELS_mat = facevalue * NN(TXY).reshape(N, N)
        ELS_mat = ELS_mat.cpu().numpy()

        plot_2d(XXX, YYY, ELS_mat, uname)

