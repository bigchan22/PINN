import numpy as np
import torch

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
import torch.nn as nn

import time
import matplotlib.pyplot as plt
import matplotlib as mat
from pylab import cm

sig_spx = 0.286  # 자산1의 변동성
sig_nvda = 0.547  # 자산2의 변동성
LV2_x = sig_spx ** 2
LV2_y = sig_nvda ** 2
LV_x = sig_spx
LV_y = sig_nvda
# d_x = 0.0128
# d_y = 0.0007
d_x = 0.
d_y = 0.
r = 0.0045  # 무위험이자율
corr = 0.5371  # 상관계수


def generate_data_block(N_coll, xbound=None, ybound=None, tbound=None):
    t_coll = torch.rand([N_coll, 1], dtype=torch.float32, device=device) * (tbound[1] - tbound[0]) + tbound[0]
    x_coll = torch.rand([N_coll, 1], dtype=torch.float32, device=device) * (xbound[1] - xbound[0]) + xbound[0]
    y_coll = torch.rand([N_coll, 1], dtype=torch.float32, device=device) * (ybound[1] - ybound[0]) + ybound[0]
    # y_coll = torch.tensor(y_coll, requires_grad=True, device=device)

    # coll = torch.cat([t_coll, x_coll, y_coll], 1).requires_grad_(True)
    coll = torch.cat([t_coll, x_coll, y_coll], 1)
    return coll


def generate_data_coll(L, T, N_coll, xbound=None, ybound=None, tbound=None):
    if tbound is None:
        t_coll = torch.rand([N_coll, 1], dtype=torch.float32, device=device) * T
    else:
        t_coll = torch.rand([N_coll, 1], dtype=torch.float32, device=device) * (tbound[1] - tbound[0]) + tbound[0]
    # t_coll = torch.tensor(t_coll, requires_grad=True, device=device)

    if xbound is None:
        x_coll = torch.rand([N_coll, 1], dtype=torch.float32, device=device) * L
    else:
        x_coll = torch.rand([N_coll, 1], dtype=torch.float32, device=device) * (xbound[1] - xbound[0]) + xbound[0]
    # x_coll = torch.tensor(x_coll, requires_grad=True, device=device)

    if ybound is None:
        y_coll = torch.rand([N_coll, 1], dtype=torch.float32, device=device) * L
    else:
        y_coll = torch.rand([N_coll, 1], dtype=torch.float32, device=device) * (ybound[1] - ybound[0]) + ybound[0]
    # y_coll = torch.tensor(y_coll, requires_grad=True, device=device)

    # coll = torch.cat([t_coll, x_coll, y_coll], 1).requires_grad_(True)
    coll = torch.cat([t_coll, x_coll, y_coll], 1)
    return coll


def generate_data_initial(L, N, xbound=None, ybound=None, tbound=None):
    t_ic = torch.zeros((N, 1), dtype=torch.float32, device=device) + tbound[0]
    if xbound is None:
        x_ic = torch.rand([N, 1], dtype=torch.float32, device=device) * L
    else:
        x_ic = torch.rand([N, 1], dtype=torch.float32, device=device) * (xbound[1] - xbound[0]) + xbound[0]
    if ybound is None:
        y_ic = torch.rand([N, 1], dtype=torch.float32, device=device) * L
    else:
        y_ic = torch.rand([N, 1], dtype=torch.float32, device=device) * (ybound[1] - ybound[0]) + ybound[0]
    ic = torch.cat([t_ic, x_ic, y_ic], 1).requires_grad_(True)
    return ic


def generate_data_boundary(T, L, N, x_or_y, u_or_l, lower_bound=0, t_lower_bound=0):
    t_b = torch.rand([N, 1], dtype=torch.float32, device=device) * (T - t_lower_bound) + t_lower_bound
    if x_or_y == "x":
        y_b = torch.rand([N, 1], dtype=torch.float32, device=device) * (L - lower_bound) + lower_bound
        if u_or_l == 'l':
            x_b = torch.zeros((N, 1), dtype=torch.float32, device=device) + lower_bound
        elif u_or_l == 'u':
            x_b = torch.zeros((N, 1), dtype=torch.float32, device=device) + L  ### +L을 더해야하는 것?
        else:
            raise ValueError
    elif x_or_y == "y":
        x_b = torch.rand([N, 1], dtype=torch.float32, device=device) * (L - lower_bound) + lower_bound
        if u_or_l == 'l':
            y_b = torch.zeros((N, 1), dtype=torch.float32, device=device) + lower_bound
        elif u_or_l == 'u':
            y_b = torch.zeros((N, 1), dtype=torch.float32, device=device) + L  ### +L을 더해야하는 것?
        else:
            raise ValueError
    else:
        raise ValueError

    bound = torch.cat([t_b, x_b, y_b], 1).requires_grad_(True)
    return bound


def generate_data_ac(t_step, N, L, strike):
    t_ac = t_step * torch.ones(N, 1, dtype=torch.float32, device=device)
    x_ac = torch.rand([N, 1], dtype=torch.float32, device=device)
    x_ac = (L - strike) * x_ac + strike
    y_ac = torch.rand([N, 1], dtype=torch.float32, device=device)  ############# 무슨 조건인가요?
    y_ac = (L - strike) * y_ac + strike  #################?이상합니다.
    ac = torch.cat([t_ac, x_ac, y_ac], 1).requires_grad_(True)
    return ac


def Payoff(xx, yy, kib, coupon):
    grad_crit = 2500.
    # grad_crit = 0.85/0.45
    grad = (1. + coupon - torch.min(xx, yy)) / (kib - torch.min(xx, yy))
    if torch.min(xx, yy) >= kib:
        sol = (1. + coupon)
    elif grad > grad_crit:
        sol = (1. + coupon) - grad_crit * (kib - torch.min(xx, yy))
    else:
        sol = torch.min(xx, yy)
    return sol
def Payoff_x(xx, yy, kib, coupon):
    grad_crit = 2500.
    # grad_crit = 0.85/0.05
    grad = (1. + coupon - torch.min(xx, yy)) / (kib - torch.min(xx, yy))
    if torch.min(xx, yy) >= kib:
        sol = 0.
    elif grad > grad_crit:
        sol = grad_crit
    else:
        sol = 1.
    return sol
# def Payoff_xx(xx, yy, kib, coupon):
#     grad_crit = 25.
#     # grad_crit = 0.85/0.05
#     grad = (1. + coupon - torch.min(xx, yy)) / (kib - torch.min(xx, yy))
#     if torch.min(xx, yy) >= kib:
#         sol = 0.
#     elif grad > grad_crit:
#         sol = (1. + coupon) - grad_crit * (kib - torch.min(xx, yy))
#     else:
#         sol = 0.
#     return sol




def ac_sol_nn(xx, yy, nn_output, barrier, coupon):
    grad_crit = 25.
    # grad_crit = 0.85 / 0.05
    grad = (1. + coupon - nn_output) / (barrier - torch.min(xx, yy))
    if torch.min(xx, yy) >= barrier:
        sol = (1. + coupon)
    elif grad > grad_crit:
        sol = (1. + coupon) - grad_crit * (barrier - torch.min(xx, yy))
    else:
        sol = nn_output
    return sol


def generate_data_step(T, L, Ns, xbound=None, ybound=None, tbound=None):
    N_coll, N_ic, N_b = Ns
    N_sam = N_coll + N_ic + np.sum(N_b)

    N_lb_x = N_b[0]
    N_lb_y = N_b[1]
    N_ub_x = N_b[2]
    N_ub_y = N_b[3]

    # coll = generate_data_coll(L, T, N_coll, xbound, ybound, tbound)
    # Testing shock data
    coll = generate_data_coll(L, T, N_coll // 2, xbound, ybound, tbound)
    shock1 = generate_data_block(N_coll // 8, (0.4, 0.47), (0, L), tbound)
    shock2 = generate_data_block(N_coll // 8, (0, L), (0.4, 0.47), tbound)
    shock3 = generate_data_block(N_coll // 8, (0, 0.45), (0, L), tbound)
    shock4 = generate_data_block(N_coll // 8, (0, L), (0, 0.45), tbound)
    coll = torch.cat([coll, shock1, shock2, shock3,shock4], 0).requires_grad_(True)
    ###############################
    ic = generate_data_initial(L, N_ic//2, xbound, ybound, tbound)
    icshock1 = generate_data_initial(L,N_ic // 4, (0.4, 0.47), (0, L), tbound)
    icshock2 = generate_data_initial(L,N_ic // 4, (0, L), (0.4, 0.47), tbound)
    ic = torch.cat([ic, icshock1, icshock2], 0).requires_grad_(True)


    lb_x = generate_data_boundary(T, L, N_lb_x, 'x', 'l')
    lb_y = generate_data_boundary(T, L, N_lb_y, 'y', 'l')
    ub_x = generate_data_boundary(T, L, N_ub_x, 'x', 'u')
    ub_y = generate_data_boundary(T, L, N_ub_y, 'y', 'u')

    txy = torch.cat([coll,
                     ic, lb_x, lb_y, ub_x, ub_y], 0).requires_grad_(True)
    x = txy[:, 1].reshape((N_sam, 1)).requires_grad_(True)
    y = txy[:, 2].reshape((N_sam, 1)).requires_grad_(True)

    # x = txy[:, 1].reshape((N_sam, 1)).requires_grad_(True)
    # y = txy[:, 2].reshape((N_sam, 1)).requires_grad_(True)

    # txy_ki_x = txy[txy[:, 1] <= kib]  # x < kib 인 경우
    # txy_ki_y = txy[txy[:, 2] <= kib]  # y < kib 인 경우
    # txy_ki = torch.cat([txy_ki_x, txy_ki_y], 0)
    # # txy_ki = torch.cat([txy_ki_x, txy_ki_y], 0).requires_grad_(True)
    # txy_ki = txy_ki.clone().detach()
    return txy, ic, lb_x, lb_y, ub_x, ub_y, x, y


def generate_sol_step(Ns, ic, coupon, barrier, init=True, nn_pre=None, ki=True):
    N_coll, N_ic, N_b = Ns
    N_sam = N_coll + N_ic + np.sum(N_b)
    N_lb_x = N_b[0]
    N_lb_y = N_b[1]
    N_ub_x = N_b[2]
    N_ub_y = N_b[3]

    pde_sol = torch.zeros((N_sam, 1), dtype=torch.float32).to(device)
    ic_sol = torch.zeros((N_ic, 1), dtype=torch.float32).to(device)  # ic_sol, kic_sol initialized
    ic_sol_x = torch.zeros((N_ic, 1), dtype=torch.float32).to(device)  # ic_sol, kic_sol initialized
    if nn_pre is not None:
        nn_output = nn_pre(ic)
        nn_output = nn_output.clone().detach()
    for i, elem in enumerate(ic):
        xx = elem[1]
        yy = elem[2]
        if init:
            ic_sol[i] = Payoff(xx, yy, barrier, coupon)
            # ic_sol_x[i] = Payoff_x(xx, yy, barrier, coupon)
        else:
            ic_sol[i] = ac_sol_nn(xx, yy, nn_output[i], barrier, coupon)

    lb_x_sol = torch.zeros((N_lb_x, 1), dtype=torch.float32).to(device)  # x = 0
    lb_y_sol = torch.zeros((N_lb_y, 1), dtype=torch.float32).to(device)  # y = 0
    ub_x_sol = torch.zeros((N_ub_x, 1), dtype=torch.float32).to(device)  # x = L
    ub_y_sol = torch.zeros((N_ub_y, 1), dtype=torch.float32).to(device)  # y = L
    return pde_sol, ic_sol, lb_x_sol, lb_y_sol, ub_x_sol, ub_y_sol


def generate_sol(Ns, ic, coupon, kib, Strike):
    N_coll, N_ic, N_ac, N_b = Ns
    N_sam = N_coll + N_ic + np.sum(N_ac) + np.sum(N_b)
    N_ac_4 = N_ac[4]
    N_ac_3 = N_ac[3]
    N_ac_2 = N_ac[2]
    N_ac_1 = N_ac[1]
    N_ac_0 = N_ac[0]
    N_lb_x = N_b[0]
    N_lb_y = N_b[1]
    N_ub_x = N_b[2]
    N_ub_y = N_b[3]

    pde_sol = torch.zeros((N_sam, 1), dtype=torch.float32)

    # maturity, initial condition: t==step[5]
    ic_sol = torch.zeros((N_ic, 1), dtype=torch.float32)  # ic_sol, kic_sol initialized
    kic_sol = torch.zeros((N_ic, 1), dtype=torch.float32)

    for i, elem in enumerate(ic):
        #     tt = elem[0],
        xx = elem[1]
        yy = elem[2]

        if torch.min(xx, yy) < kib:
            ic_sol[i] = torch.minimum(xx, yy)
            kic_sol[i] = torch.minimum(xx, yy)

        elif torch.min(xx, yy) < (kib + 0.01):
            ic_sol[i] = (((1. + coupon[5]) - kib) / 0.01) * (torch.minimum(xx, yy) - kib) + kib
            kic_sol[i] = torch.minimum(xx, yy)

        elif torch.min(xx, yy) < Strike[5]:
            ic_sol[i] = (1. + coupon[5])
            kic_sol[i] = torch.minimum(xx, yy)

        elif torch.min(xx, yy) < (Strike[5] + 0.01):
            ic_sol[i] = (1. + coupon[5])
            kic_sol[i] = (((1. + coupon[5]) - Strike[5]) / 0.01) * (torch.minimum(xx, yy) - Strike[5]) \
                         + Strike[5]

        else:
            ic_sol[i] = (1. + coupon[5])
            kic_sol[i] = (1. + coupon[5])

    # auto call early redemption t ==step[4] 0.5y
    ac_4_sol = torch.ones((N_ac_4, 1), dtype=torch.float32) * (1. + coupon[4])

    # auto call early redemption t ==step[3] 1y
    ac_3_sol = torch.ones((N_ac_3, 1), dtype=torch.float32) * (1. + coupon[3])

    # auto call early redemption t ==step[2] 1.5y
    ac_2_sol = torch.ones((N_ac_2, 1), dtype=torch.float32) * (1. + coupon[2])

    # auto call early redemption t ==step[1] 2y
    ac_1_sol = torch.ones((N_ac_1, 1), dtype=torch.float32) * (1. + coupon[1])

    # auto call early redemption t ==step[0] 2.5y
    ac_0_sol = torch.ones((N_ac_0, 1), dtype=torch.float32) * (1. + coupon[0])

    # boundaries (LB: Dirichlet Boundary condition, UB: Neumann Boundary condition (Zero Gamma))
    # x  = 0
    lb_x_sol = torch.zeros((N_lb_x, 1), dtype=torch.float32)

    # y = 0
    lb_y_sol = torch.zeros((N_lb_y, 1), dtype=torch.float32)

    # x = L
    ub_x_sol = torch.zeros((N_ub_x, 1), dtype=torch.float32)

    # y = L
    ub_y_sol = torch.zeros((N_ub_y, 1), dtype=torch.float32)

    pde_sol = torch.tensor(pde_sol).reshape(N_sam, 1).to(device)
    ic_sol = torch.tensor(ic_sol).reshape(N_ic, 1).to(device)
    kic_sol = torch.tensor(kic_sol).reshape(N_ic, 1).to(device)

    ac_4_sol = torch.tensor(ac_4_sol).reshape(N_ac_4, 1).to(device)
    ac_3_sol = torch.tensor(ac_3_sol).reshape(N_ac_3, 1).to(device)
    ac_2_sol = torch.tensor(ac_2_sol).reshape(N_ac_2, 1).to(device)
    ac_1_sol = torch.tensor(ac_1_sol).reshape(N_ac_1, 1).to(device)
    ac_0_sol = torch.tensor(ac_0_sol).reshape(N_ac_0, 1).to(device)

    lb_x_sol = torch.tensor(lb_x_sol).reshape(N_lb_x, 1).to(device)
    lb_y_sol = torch.tensor(lb_y_sol).reshape(N_lb_y, 1).to(device)
    ub_x_sol = torch.tensor(ub_x_sol).reshape(N_ub_x, 1).to(device)
    ub_y_sol = torch.tensor(ub_y_sol).reshape(N_ub_y, 1).to(device)

    return pde_sol, ic_sol, kic_sol, ac_4_sol, ac_3_sol, ac_2_sol, ac_1_sol, ac_0_sol, \
           lb_x_sol, lb_y_sol, ub_x_sol, ub_y_sol


# Random Spatial numbers Loader
def generate_data_bspde_gbm_put_3param_ELS(T, L, r, Ns, kib, coupon, step, Strike):
    N_coll, N_ic, N_ac, N_b = Ns
    N_sam = N_coll + N_ic + np.sum(N_ac) + np.sum(N_b)
    N_ac_4 = N_ac[4]
    N_ac_3 = N_ac[3]
    N_ac_2 = N_ac[2]
    N_ac_1 = N_ac[1]
    N_ac_0 = N_ac[0]
    N_lb_x = N_b[0]
    N_lb_y = N_b[1]
    N_ub_x = N_b[2]
    N_ub_y = N_b[3]

    coll = generate_data_coll(L, T, N_coll)
    ic = generate_data_initial(L, N_ic)

    ac_4 = generate_data_ac(step[4], N_ac_4, L, Strike[4])
    ac_3 = generate_data_ac(step[3], N_ac_3, L, Strike[3])
    ac_2 = generate_data_ac(step[2], N_ac_2, L, Strike[2])
    ac_1 = generate_data_ac(step[1], N_ac_1, L, Strike[1])
    ac_0 = generate_data_ac(step[0], N_ac_0, L, Strike[0])

    lb_x = generate_data_boundary(T, L, N_lb_x, 'x', 'l')
    lb_y = generate_data_boundary(T, L, N_lb_y, 'y', 'l')
    ub_x = generate_data_boundary(T, L, N_ub_x, 'x', 'u')
    ub_y = generate_data_boundary(T, L, N_ub_y, 'y', 'u')

    txy = torch.cat([coll,
                     ic, ac_4, ac_3, ac_2, ac_1, ac_0,
                     lb_x, lb_y, ub_x, ub_y], 0).requires_grad_(True)

    x = txy[:, 1].reshape((N_sam, 1)).requires_grad_(True)
    y = txy[:, 2].reshape((N_sam, 1)).requires_grad_(True)

    txy_ki_x = txy[txy[:, 1] <= kib]  # x < kib 인 경우
    txy_ki_y = txy[txy[:, 2] <= kib]  # y < kib 인 경우
    txy_ki = torch.cat([txy_ki_x, txy_ki_y], 0)
    # txy_ki = torch.cat([txy_ki_x, txy_ki_y], 0).requires_grad_(True)
    txy_ki = txy_ki.clone().detach()
    #    txy_ki = torch.unique(txy_ki, axis=0)
    #    txy_ki = torch.unique(txy_ki)

    pde_sol, ic_sol, kic_sol, ac_4_sol, ac_3_sol, ac_2_sol, ac_1_sol, ac_0_sol, \
    lb_x_sol, lb_y_sol, ub_x_sol, ub_y_sol \
        = generate_sol(Ns, ic, coupon, kib, Strike)

    return x, y, txy, ic, ac_4, ac_3, ac_2, ac_1, ac_0, lb_x, lb_y, ub_x, ub_y, ic_sol, kic_sol, \
           ac_4_sol, ac_3_sol, ac_2_sol, ac_1_sol, ac_0_sol, \
           lb_x_sol, lb_y_sol, ub_x_sol, ub_y_sol, pde_sol, txy_ki, \
        #             txy_zero


def output_PINN(nn, txy, ic, lb_x, lb_y, ub_x, ub_y):
    u = nn(txy)
    u_1 = torch.autograd.grad(inputs=txy, outputs=u, grad_outputs=torch.ones_like(u), retain_graph=True,
                              create_graph=True)[0]
    u_t = u_1[:, 0].reshape(-1, 1)
    u_x = u_1[:, 1].reshape(-1, 1)
    u_y = u_1[:, 2].reshape(-1, 1)
    u_2 = torch.autograd.grad(inputs=txy, outputs=u_x, grad_outputs=torch.ones_like(u_x), retain_graph=True,
                              create_graph=True)[0]
    u_xx = u_2[:, 1].reshape(-1, 1)
    u_xy = u_2[:, 2].reshape(-1, 1)
    u_yy = torch.autograd.grad(inputs=txy, outputs=u_y, grad_outputs=torch.ones_like(u_y), retain_graph=True,
                               create_graph=True)[0][:, 2].reshape(-1, 1)

    u_ic = nn(ic)


    u_lb_x = nn(lb_x)
    u_lb_y = nn(lb_y)

    u_ub_x = nn(ub_x)
    u_ub_y = nn(ub_y)

    u_lb_x_x = torch.autograd.grad(inputs=lb_x, outputs=u_lb_x, grad_outputs=torch.ones_like(u_lb_x),
                                   retain_graph=True, create_graph=True)[0][:, 1].reshape(-1, 1)
    u_lb_x_xx = \
        torch.autograd.grad(inputs=lb_x, outputs=u_lb_x_x, grad_outputs=torch.ones_like(u_lb_x_x),
                            retain_graph=True,
                            create_graph=True)[0][:, 1].reshape(-1, 1)

    u_lb_y_y = \
        torch.autograd.grad(inputs=lb_y, outputs=u_lb_y, grad_outputs=torch.ones_like(u_lb_y), retain_graph=True,
                            create_graph=True)[0][:, 2].reshape(-1, 1)
    u_lb_y_yy = \
        torch.autograd.grad(inputs=lb_y, outputs=u_lb_y_y, grad_outputs=torch.ones_like(u_lb_y_y),
                            retain_graph=True,
                            create_graph=True)[0][:, 2].reshape(-1, 1)

    u_ub_x_x = \
        torch.autograd.grad(inputs=ub_x, outputs=u_ub_x, grad_outputs=torch.ones_like(u_ub_x), retain_graph=True,
                            create_graph=True)[0][:, 1].reshape(-1, 1)
    u_ub_x_xx = \
        torch.autograd.grad(inputs=ub_x, outputs=u_ub_x_x, grad_outputs=torch.ones_like(u_ub_x_x),
                            retain_graph=True,
                            create_graph=True)[0][:, 1].reshape(-1, 1)

    u_ub_y_y = \
        torch.autograd.grad(inputs=ub_y, outputs=u_ub_y, grad_outputs=torch.ones_like(u_ub_y), retain_graph=True,
                            create_graph=True)[0][:, 2].reshape(-1, 1)
    u_ub_y_yy = \
        torch.autograd.grad(inputs=ub_y, outputs=u_ub_y_y, grad_outputs=torch.ones_like(u_ub_y_y),
                            retain_graph=True,
                            create_graph=True)[0][:, 2].reshape(-1, 1)
    u1 = (u_t, u_x, u_y)
    u2 = (u_xx, u_xy, u_yy)
    ub = (u_lb_x, u_lb_y, u_ub_x, u_ub_y)
    ub1 = (u_lb_x_x, u_lb_y_y, u_ub_x_x, u_ub_y_y)
    ub2 = (u_lb_x_xx, u_lb_y_yy, u_ub_x_xx, u_ub_y_yy)
    us = (u, u1, u2, u_ic, ub, ub1, ub2)
    return us


def Loss_PINN(us, Ns, ic, x, y, cost_f, coupon, barrier, init=True, nn_pre=None):
    u, u1, u2, u_ic, ub, ub1, ub2 = us
    u_t, u_x, u_y = u1
    u_xx, u_xy, u_yy = u2
    u_lb_x, u_lb_y, u_ub_x, u_ub_y = ub
    u_lb_x_x, u_lb_y_y, u_ub_x_x, u_ub_y_y = ub1
    u_lb_x_xx, u_lb_y_yy, u_ub_x_xx, u_ub_y_yy = ub2
    pde_sol, ic_sol, lb_x_sol, lb_y_sol, ub_x_sol, ub_y_sol = \
        generate_sol_step(Ns, ic, coupon, barrier, init=init, nn_pre=nn_pre)

    pde_u = u_t - 0.5 * LV2_x * x ** 2 * u_xx - 0.5 * LV2_y * y ** 2 * u_yy \
            - corr * LV_x * LV_y * x * y * u_xy \
            - (r - d_x) * x * u_x - (r - d_y) * y * u_y + r * u

    loss_pde_u = cost_f(pde_u, pde_sol)
    loss_ic_u = cost_f(u_ic, ic_sol)

    loss_lb_x_u = cost_f(u_lb_x, lb_x_sol)
    loss_lb_y_u = cost_f(u_lb_y, lb_y_sol)
    loss_ub_x_u = cost_f(u_ub_x_x, ub_x_sol)
    loss_ub_y_u = cost_f(u_ub_y_y, ub_y_sol)
    loss_u = 1 * loss_pde_u + 10 * loss_ic_u \
             + 10 * (loss_lb_x_u + loss_lb_y_u + loss_ub_x_u + loss_ub_y_u)
    return loss_u
