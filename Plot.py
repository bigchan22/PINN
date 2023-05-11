import time

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib as mat
from pylab import cm
#
# x_len_u = torch.arange(len(loss_history_u))
# x_len_ku = torch.arange(len(loss_history_ku))
#
# plt.subplot(211)
# plt.plot(x_len_u, loss_history_u, marker='.', c='blue', label="Training Loss for u")
# plt.legend(loc='upper right')
# plt.grid()
# plt.xlabel('Iteration')
# plt.ylabel('loss')
#
# plt.subplot(212)
# plt.plot(x_len_ku, loss_history_ku, marker='.', c='blue', label="Training Loss for ku")
# plt.legend(loc='upper right')
# plt.grid()
# plt.xlabel('Iteration')
# plt.ylabel('loss')
#
# plt.show()


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
    plt.savefig(time.strftime('./Training_Loss_u%Y-%m-%d-%H-%M-%S' + lossname), dpi=200)


# u_ic

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
    ax.set_title('ELS Price')
    #     fig.savefig('PINN_u_maturity.png')
    plt.savefig('./' + uname + time.strftime('%Y-%m-%d-%H-%M-%S'), dpi=200)


# def plot_ELS(T, L, N, device, facevalue, NN, uname):
#     with torch.autograd.no_grad():
#         ONES = torch.ones((N ** 2, 1))
#         # ZEROS = torch.zeros((121 ** 2, 1))
#         xxx = torch.linspace(0., L, N).reshape(N, 1) / L * L * 1
#         yyy = xxx
#         XXX, YYY = torch.meshgrid(xxx, yyy)
#         XXX_flat = XXX.flatten()[:, None]
#         YYY_flat = YYY.flatten()[:, None]
#         XY = torch.cat((XXX_flat, YYY_flat), axis=1)
#         TXY = torch.cat((T * ONES, XY), axis=1)
#         TXY = torch.FloatTensor(TXY).to(device)
#
#         ELS_mat = facevalue * NN(TXY).reshape(L, L)
#         ELS_mat = ELS_mat.cpu().numpy()
#
#         plot_2d(XXX, YYY, ELS_mat, uname)
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


#
#
# plot_ELS(step[5], L, 121, device, facevalue, NN5, "PINN_u_maturity")
# plot_ELS(T, L, 121, device, facevalue, NN5, "PINN_u_maturity")
#
#
# with torch.autograd.no_grad():
#     u_outcome = torch.tensor([T, 1.0 / L * L * 1, 1.0 / L * L * 1]).reshape((1, 3))
#     u_outcome = torch.FloatTensor(u_outcome).to(device)
#     print('u price : {}'.format(NN5(u_outcome) * facevalue))
#
#     ku_outcome = torch.tensor([T, 1.0 / L * L * 1, 1.0 / L * L * 1]).reshape((1, 3))
#     ku_outcome = torch.FloatTensor(ku_outcome).to(device)
#     print('ku price : {}'.format(NN6(ku_outcome) * facevalue))
#
# # with torch.autograd.no_grad():
# ONES = torch.ones((121 ** 2, 1))
# ZEROS = torch.zeros((121 ** 2, 1))
# xxx = torch.linspace(0., L, 121).reshape(121, 1) / L * L * 1
# yyy = xxx
# XXX, YYY = torch.meshgrid(xxx, yyy)
# XXX_flat = XXX.flatten()[:, None]
# YYY_flat = YYY.flatten()[:, None]
# XY = torch.cat((XXX_flat, YYY_flat), axis=1)
# TXY = torch.cat((T * ONES, XY), axis=1)
# TXY = torch.FloatTensor(TXY).to(device)
# TXY.requires_grad = True
#
# uu = NN5(TXY)
# uu_1 = torch.autograd.grad(inputs=TXY, outputs=uu, grad_outputs=torch.ones_like(uu),
#                            retain_graph=True, create_graph=True)[0]
# uu_t = uu_1[:, 0].reshape(-1, 1)
# uu_x = uu_1[:, 1].reshape(-1, 1)
# uu_y = uu_1[:, 2].reshape(-1, 1)
# uu_2 = torch.autograd.grad(inputs=TXY, outputs=uu_x, grad_outputs=torch.ones_like(uu_x),
#                            retain_graph=True, create_graph=True)[0]
# uu_xx = uu_2[:, 1].reshape(-1, 1)
# uu_xy = uu_2[:, 2].reshape(-1, 1)
# uu_yy = torch.autograd.grad(inputs=TXY, outputs=uu_y, grad_outputs=torch.ones_like(uu_y),
#                             retain_graph=True, create_graph=True)[0][:, 2].reshape(-1, 1)
#
# delta_x_mat = uu_x.reshape(121, 121)
# delta_y_mat = uu_y.reshape(121, 121)
# gamma_x_mat = uu_xx.reshape(121, 121)
# gamma_y_mat = uu_yy.reshape(121, 121)
# crossgamma_xy_mat = uu_xy.reshape(121, 121)
# theta_mat = uu_t.reshape(121, 121)
#
# # u delta_x
# with torch.autograd.no_grad():
#     fig = plt.figure(figsize=(10, 10))
#     ax = fig.add_subplot(111, projection='3d')
#     surf = ax.plot_surface(XXX, YYY, delta_x_mat.cpu().numpy(), cmap=cm.coolwarm)
#
#     ax.view_init(20, -135)
#     ax.set_xlabel('S1', fontsize=12)
#     ax.set_ylabel('S2', fontsize=12)
#
#     ax.set_xticks(torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]))
#     ax.set_xticklabels(['$0$', '$50$', '$100$', '$150$', '$200$', '$250$', '$300$'])
#     ax.set_yticks(torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]))
#     ax.set_yticklabels(['$0$', '$50$', '$100$', '$150$', '$200$', '$250$', '$300$'])
#     ax.set_title('Delta 1')
#     #     fig.savefig('PINN_u_delta_1.png')
#     plt.savefig(time.strftime('Pictures/ELS/torch/PINN_u_delta_1%Y-%m-%d-%H-%M-%S'), dpi=200)
#
# # u delta_y
# with torch.autograd.no_grad():
#     fig = plt.figure(figsize=(10, 10))
#     ax = fig.add_subplot(111, projection='3d')
#     surf = ax.plot_surface(XXX, YYY, delta_y_mat.cpu().numpy(), cmap=cm.coolwarm)
#
#     ax.view_init(20, -135)
#     ax.set_xlabel('S1', fontsize=12)
#     ax.set_ylabel('S2', fontsize=12)
#
#     ax.set_xticks(torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]))
#     ax.set_xticklabels(['$0$', '$50$', '$100$', '$150$', '$200$', '$250$', '$300$'])
#     ax.set_yticks(torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]))
#     ax.set_yticklabels(['$0$', '$50$', '$100$', '$150$', '$200$', '$250$', '$300$'])
#     ax.set_title('Delta 2')
#     #     fig.savefig('PINN_u_delta_2.png')
#     plt.savefig(time.strftime('Pictures/ELS/torch/PINN_u_delta_2_%Y-%m-%d-%H-%M-%S'), dpi=200)
#
# # u gamma_x
# with torch.autograd.no_grad():
#     fig = plt.figure(figsize=(10, 10))
#     ax = fig.add_subplot(111, projection='3d')
#     surf = ax.plot_surface(XXX, YYY, gamma_x_mat.cpu().numpy(), cmap=cm.coolwarm)
#
#     ax.view_init(20, -135)
#     ax.set_xlabel('S1', fontsize=12)
#     ax.set_ylabel('S2', fontsize=12)
#
#     ax.set_xticks(torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]))
#     ax.set_xticklabels(['$0$', '$50$', '$100$', '$150$', '$200$', '$250$', '$300$'])
#     ax.set_yticks(torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]))
#     ax.set_yticklabels(['$0$', '$50$', '$100$', '$150$', '$200$', '$250$', '$300$'])
#     ax.set_title('Gamma 1')
#     #     fig.savefig('PINN_u_gamma_1.png')
#     plt.savefig(time.strftime('Pictures/ELS/torch/PINN_u_gamma_1_%Y-%m-%d-%H-%M-%S'), dpi=200)
#
# # u gamma_y
# with torch.autograd.no_grad():
#     fig = plt.figure(figsize=(10, 10))
#     ax = fig.add_subplot(111, projection='3d')
#     surf = ax.plot_surface(XXX, YYY, gamma_y_mat.cpu().numpy(), cmap=cm.coolwarm)
#
#     ax.view_init(20, -135)
#     ax.set_xlabel('S1', fontsize=12)
#     ax.set_ylabel('S2', fontsize=12)
#
#     ax.set_xticks(torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]))
#     ax.set_xticklabels(['$0$', '$50$', '$100$', '$150$', '$200$', '$250$', '$300$'])
#     ax.set_yticks(torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]))
#     ax.set_yticklabels(['$0$', '$50$', '$100$', '$150$', '$200$', '$250$', '$300$'])
#     ax.set_title('Gamma 2')
#     #     fig.savefig('PINN_u_gamma_2.png')
#     plt.savefig(time.strftime('Pictures/ELS/torch/PINN_u_gamma_2_%Y-%m-%d-%H-%M-%S'), dpi=200)
#
# # u crossgamma
# # u gamma_y
# with torch.autograd.no_grad():
#     fig = plt.figure(figsize=(10, 10))
#     ax = fig.add_subplot(111, projection='3d')
#     surf = ax.plot_surface(XXX, YYY, crossgamma_xy_mat.cpu().numpy(), cmap=cm.coolwarm)
#
#     ax.view_init(20, -135)
#     ax.set_xlabel('S1', fontsize=12)
#     ax.set_ylabel('S2', fontsize=12)
#
#     ax.set_xticks(torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]))
#     ax.set_xticklabels(['$0$', '$50$', '$100$', '$150$', '$200$', '$250$', '$300$'])
#     ax.set_yticks(torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]))
#     ax.set_yticklabels(['$0$', '$50$', '$100$', '$150$', '$200$', '$250$', '$300$'])
#     ax.set_title('Cross gamma')
#     #     fig.savefig('PINN_u_crossgamma.png')
#     plt.savefig(time.strftime('Pictures/ELS/torch/PINN_u_crossgamma%Y-%m-%d-%H-%M-%S'), dpi=200)
#
# # u theta
# with torch.autograd.no_grad():
#     fig = plt.figure(figsize=(10, 10))
#     ax = fig.add_subplot(111, projection='3d')
#     surf = ax.plot_surface(XXX, YYY, -theta_mat.cpu().numpy(), cmap=cm.coolwarm)
#
#     ax.view_init(20, -135)
#     ax.set_xlabel('S1', fontsize=12)
#     ax.set_ylabel('S2', fontsize=12)
#
#     ax.set_xticks(torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]))
#     ax.set_xticklabels(['$0$', '$50$', '$100$', '$150$', '$200$', '$250$', '$300$'])
#     ax.set_yticks(torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]))
#     ax.set_yticklabels(['$0$', '$50$', '$100$', '$150$', '$200$', '$250$', '$300$'])
#     ax.set_title('Theta')
#     #     fig.savefig('PINN_u_theta.png')
#     plt.savefig(time.strftime('Pictures/ELS/torch/PINN_u_theta%Y-%m-%d-%H-%M-%S'), dpi=200)
