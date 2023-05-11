import numpy as np
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import torch.nn as nn

import time
import matplotlib.pyplot as plt
import matplotlib as mat
from pylab import cm


def generate_data_coll(L, T, N_coll):
    t_coll = torch.rand([N_coll, 1], dtype=torch.float32, device=device) * T
    t_coll = torch.tensor(t_coll, requires_grad=True, device=device)
    x_coll = torch.rand([N_coll, 1], dtype=torch.float32, device=device) * L
    x_coll = torch.tensor(x_coll, requires_grad=True, device=device)
    y_coll = torch.rand([N_coll, 1], dtype=torch.float32, device=device) * L
    y_coll = torch.tensor(y_coll, requires_grad=True, device=device)
    coll = torch.cat([t_coll, x_coll, y_coll], 1).requires_grad_(True)
    return coll


def generate_data_initial(L, N):
    t_ic = torch.zeros((N, 1), dtype=torch.float32, device=device)
    x_ic = torch.rand([N, 1], dtype=torch.float32, device=device) * L
    y_ic = torch.rand([N, 1], dtype=torch.float32, device=device) * L
    ic = torch.cat([t_ic, x_ic, y_ic], 1).requires_grad_(True)
    return ic


def generate_data_boundary(T, L, N, x_or_y, u_or_l):
    t_b = torch.rand([N, 1], dtype=torch.float32, device=device) * T
    if x_or_y == "x":
        y_b = torch.rand([N, 1], dtype=torch.float32, device=device) * L
        if u_or_l == 'l':
            x_b = torch.zeros((N, 1), dtype=torch.float32, device=device)
        elif u_or_l == 'u':
            x_b = torch.zeros((N, 1), dtype=torch.float32, device=device) + L  ### +L을 더해야하는 것?
        else:
            raise ValueError
    elif x_or_y == "y":
        x_b = torch.rand([N, 1], dtype=torch.float32, device=device) * L
        if u_or_l == 'l':
            y_b = torch.zeros((N, 1), dtype=torch.float32, device=device)
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
    txy_ki = torch.cat([txy_ki_x, txy_ki_y], 0).requires_grad_(True)
    #    txy_ki = torch.unique(txy_ki, axis=0)
    #    txy_ki = torch.unique(txy_ki)

    pde_sol, ic_sol, kic_sol, ac_4_sol, ac_3_sol, ac_2_sol, ac_1_sol, ac_0_sol, \
    lb_x_sol, lb_y_sol, ub_x_sol, ub_y_sol \
        = generate_sol(Ns, ic, coupon, kib, Strike)

    return x, y, txy, ic, ac_4, ac_3, ac_2, ac_1, ac_0, lb_x, lb_y, ub_x, ub_y, ic_sol, kic_sol, \
           ac_4_sol, ac_3_sol, ac_2_sol, ac_1_sol, ac_0_sol, \
           lb_x_sol, lb_y_sol, ub_x_sol, ub_y_sol, pde_sol, txy_ki, \
        #             txy_zero
