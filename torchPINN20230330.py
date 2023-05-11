#!/usr/bin/env python
# coding: utf-8
# torchPINN20230328을 완전 torch로 바꾸자 2023.03.30 힘드네
import torch
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib as mat
from pylab import cm

# T = 1097./365. # 2021-09-08 ~ 2024-09-09 : 1097 days / 3 years
T = 1088. / 365.  # 2021-09-17 ~ 2024-09-09 : 1091 days / 3 years
# div = 0.0128
r = 0.0045  # 무위험이자율
# L = 2.0
L = 3.0  # 자산 최대 크기 영역
sig_spx = 0.286  # 자산1의 변동성
sig_nvda = 0.547  # 자산2의 변동성
coupon = [0.091 * 0.5, 0.091 * 1.0, 0.091 * 1.5, 0.091 * 2.0, 0.091 * 2.5,
          0.091 * 3.0]  # 잔존기별 쿠폰 2.5y 2y 1.5y 1y 0.5y 0y
Strike = [0.85, 0.85, 0.85, 0.80, 0.80, 0.70]  # 잔존만기별 행사가격 2.5y 2y 1.5y 1y 0.5y 0y
kib = 0.45  # knock in barrier
corr = 0.5371  # 상관계수
facevalue = 100  # 액면가
step = [910. / 365, 727. / 365, 545. / 365, 363. / 365, 184. / 365, 0.]  # 잔존만기 2.5y 2y 1.5y 1y 0.5y 0y
# step = [184./365 , 363/365, 545./365, 727./365, 910./365, T] 
LV2_x = sig_spx ** 2
LV2_y = sig_nvda ** 2
LV_x = sig_spx
LV_y = sig_nvda
# d_x = 0.0128
# d_y = 0.0007
d_x = 0.
d_y = 0.

# We consider Net as our solution u_theta(t, x,y)

"""
When forming the network, we have to keep in mind the number of inputs and outputs
In our case: #inputs = 3 (t, x,y)
         and #outputs = 1

You can add ass many hidden layers as you want with as many neurons.
More complex the network, the more prepared it is to find complex solutions, but it also requires more data.

Let us create this network:
min 5 hidden layer with 5 neurons each.
"""


# PINN net 정의
class PINN_BS_2D(nn.Module):
    def __init__(self):
        super(PINN_BS_2D, self).__init__()
        self.net = nn.Sequential(
            #           nn.Linear(3, 256),            nn.Tanh(),
            #          nn.Linear(256, 256),          nn.Tanh(),
            #         nn.Linear(256, 256),          nn.Tanh(), #tanh는 초기값이 좋지않다 tau=-
            #        nn.Linear(256, 1),    )
            nn.Linear(3, 32), nn.Softplus(),  # nn.ReLU(), #nn.Tanh(),ReLU는 현재값이 좋지 않다.tau=T
            nn.Linear(32, 32), nn.Softplus(),  # nn.ReLU(), #nn.Tanh(),
            nn.Linear(32, 32), nn.ReLU(),  # nn.Softplus(), #nn.ReLU(), # nn.Tanh(),
            nn.Linear(32, 32), nn.Softplus(),  # nn.ReLU(), # nn.Tanh(),
            nn.Linear(32, 1))

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, x):
        return self.net(x).to(device)


### (2) Model
NN5 = PINN_BS_2D().to(device)  # ELS price for No touch kib
NN6 = PINN_BS_2D().to(device)  # ELS price for touch kib

mse_cost_function = torch.nn.MSELoss().to(device)  # Mean squared error

# Adam optimizer: lr, beta1, beta2, epsilon default setting
optimizer5 = torch.optim.Adam(NN5.parameters())
optimizer6 = torch.optim.Adam(NN6.parameters())


# L-BFGS optimizer: lr, max_iter, max_eval, tolerance_grad, tolerance_change, history_size default setting
# optimizer5 = torch.optim.LBFGS(NN5.parameters())
# optimizer6 = torch.optim.LBFGS(NN6.parameters())

# closure
def closure_u_Adam():  # u는 knock in 안 했을 때 가격
    if torch.is_grad_enabled():
        optimizer5.zero_grad()
    if loss_u.requires_grad:
        loss_u.backward()
    return loss_u


def closure_ku_Adam():  # u는 knock in 했을 때의 가
    if torch.is_grad_enabled():
        optimizer6.zero_grad()
    if loss_ku.requires_grad:
        loss_ku.backward()
    return loss_ku


# Random Spatial numbers Loader
def generate_data_bspde_gbm_put_3param_ELS(T, L, r):
    # KOREA INVESTMENT SECURITIES 14361
    # Evaluation Date: 20210908
    # Underlying Asset: NVDA, S&P500
    # Par Value: $100
    # Fair Value at Evaluation Date: $ 85.5074
    # Volatility, NVDA 54.7%, S&P500 28.6%
    # Correlation Coefficient: 0.5371
    # Start: 2021-09-17
    # Maturities: 2022-03-14, 2022-09-13, 2023-03-14, 2023-09-12, 2024-03-09, 2024-09-09
    # Time to maturity(from 20210908): [1097, 910, 727, 545, 363, 184, 0] / T_ELS
    # Coupon: 0.091 * [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    # Strikes: 0.85-0.85-0.85-0.80-0.80-0.70
    # KI barrier: 0.45
    # Payoff(Profit): (1 + Coupon[i]) * Par
    # Payoff(NO-KI, Loss): (min(S1[T]/S1[0], S2[T]/S2[0]) - 1) * Par
    # Payoff(KI, Loss): (min(S1[T]/S1[0], S2[T]/S2[0]) - 1) * Par

    N_coll = 5000  # 0 #100000 #10000 # Number of collocation points

    N_ic = 1000  # 500  #초기값 점들 수
    N_ac_4 = 100  # 조기상환 시각의 점들 수격 ac=autocall
    N_ac_3 = 100
    N_ac_2 = 100
    N_ac_1 = 100
    N_ac_0 = 100

    N_lb_x = 100  # lower bound 점들 수
    N_lb_y = 100
    N_ub_x = 100  # upper bound
    N_ub_y = 100

    N_sam = N_coll + N_ic + N_ac_4 + N_ac_3 + N_ac_2 + N_ac_1 + N_ac_0 + N_lb_x \
            + N_lb_y + N_ub_x + N_ub_y

    t_coll = torch.rand([N_coll, 1], dtype=torch.float32, device=device) * T
    t_coll = torch.tensor(t_coll, requires_grad=True, device=device)
    x_coll = torch.rand([N_coll, 1], dtype=torch.float32, device=device) * L
    x_coll = torch.tensor(x_coll, requires_grad=True, device=device)
    y_coll = torch.rand([N_coll, 1], dtype=torch.float32, device=device) * L
    y_coll = torch.tensor(y_coll, requires_grad=True, device=device)
    coll = torch.cat([t_coll, x_coll, y_coll], 1).requires_grad_(True)

    t_ic = torch.zeros((N_ic, 1), dtype=torch.float32, device=device)
    x_ic = torch.rand([N_ic, 1], dtype=torch.float32, device=device) * L
    y_ic = torch.rand([N_ic, 1], dtype=torch.float32, device=device) * L
    ic = torch.cat([t_ic, x_ic, y_ic], 1).requires_grad_(True)

    t_ac_4 = step[4] * torch.ones(N_ac_4, 1, dtype=torch.float32, device=device)
    x_ac_4 = torch.rand([N_ac_4, 1], dtype=torch.float32, device=device)
    x_ac_4 = (L - Strike[4]) * x_ac_4 + Strike[4]
    y_ac_4 = torch.rand([N_ac_4, 1], dtype=torch.float32, device=device)
    y_ac_4 = (L - Strike[4]) * x_ac_4 + Strike[4]
    ac_4 = torch.cat([t_ac_4, x_ac_4, y_ac_4], 1).requires_grad_(True)

    t_ac_3 = step[3] * torch.ones((N_ac_3, 1), dtype=torch.float32, device=device)
    x_ac_3 = torch.rand([N_ac_3, 1], dtype=torch.float32, device=device)
    x_ac_3 = (L - Strike[3]) * x_ac_3 + Strike[3]
    y_ac_3 = torch.rand([N_ac_3, 1], dtype=torch.float32, device=device)
    y_ac_3 = (L - Strike[3]) * x_ac_3 + Strike[3]
    ac_3 = torch.cat([t_ac_3, x_ac_3, y_ac_3], 1).requires_grad_(True)

    t_ac_2 = step[2] * torch.ones((N_ac_2, 1), dtype=torch.float32, device=device)
    x_ac_2 = torch.rand([N_ac_2, 1], dtype=torch.float32, device=device)
    x_ac_2 = (L - Strike[2]) * x_ac_2 + Strike[2]
    y_ac_2 = torch.rand([N_ac_2, 1], dtype=torch.float32, device=device)
    y_ac_2 = (L - Strike[2]) * x_ac_2 + Strike[2]
    ac_2 = torch.cat([t_ac_2, x_ac_2, y_ac_2], 1).requires_grad_(True)

    t_ac_1 = step[1] * torch.ones((N_ac_1, 1), dtype=torch.float32, device=device)
    x_ac_1 = torch.rand([N_ac_1, 1], dtype=torch.float32, device=device)
    x_ac_1 = (L - Strike[1]) * x_ac_1 + Strike[1]
    y_ac_1 = torch.rand([N_ac_1, 1], dtype=torch.float32, device=device)
    y_ac_1 = (L - Strike[1]) * x_ac_1 + Strike[1]
    ac_1 = torch.cat([t_ac_1, x_ac_1, y_ac_1], 1).requires_grad_(True)

    t_ac_0 = step[0] * torch.ones((N_ac_0, 1), dtype=torch.float32, device=device)
    x_ac_0 = torch.rand([N_ac_0, 1], dtype=torch.float32, device=device)
    x_ac_0 = (L - Strike[0]) * x_ac_0 + Strike[0]
    y_ac_0 = torch.rand([N_ac_0, 1], dtype=torch.float32, device=device)
    y_ac_0 = (L - Strike[2]) * x_ac_0 + Strike[0]
    ac_0 = torch.cat([t_ac_0, x_ac_0, y_ac_0], 1).requires_grad_(True)

    t_lb_x = torch.rand([N_lb_x, 1], dtype=torch.float32, device=device) * T
    x_lb_x = torch.zeros((N_lb_x, 1), dtype=torch.float32, device=device, )
    y_lb_x = torch.rand([N_lb_x, 1], dtype=torch.float32, device=device) * L
    lb_x = torch.cat([t_lb_x, x_lb_x, y_lb_x], 1).requires_grad_(True)

    t_lb_y = torch.rand([N_lb_y, 1], dtype=torch.float32, device=device) * T
    x_lb_y = torch.zeros((N_lb_y, 1), dtype=torch.float32, device=device)
    y_lb_y = torch.rand([N_lb_y, 1], dtype=torch.float32, device=device) * L
    lb_y = torch.cat([t_lb_y, x_lb_y, y_lb_y], 1).requires_grad_(True)

    t_ub_x = torch.rand([N_ub_x, 1], dtype=torch.float32, device=device) * T
    x_ub_x = torch.zeros((N_ub_x, 1), dtype=torch.float32, device=device)
    y_ub_x = torch.rand([N_ub_x, 1], dtype=torch.float32, device=device) * L
    ub_x = torch.cat([t_ub_x, x_ub_x, y_ub_x], 1).requires_grad_(True)

    t_ub_y = torch.rand([N_ub_y, 1], dtype=torch.float32, device=device) * T
    x_ub_y = torch.zeros((N_ub_y, 1), dtype=torch.float32, device=device)
    y_ub_y = torch.rand([N_ub_y, 1], dtype=torch.float32, device=device) * L
    ub_y = torch.cat([t_ub_y, x_ub_y, y_ub_y], 1).requires_grad_(True)

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

    return x, y, txy, ic, ac_4, ac_3, ac_2, ac_1, ac_0, lb_x, lb_y, ub_x, ub_y, ic_sol, kic_sol, \
           ac_4_sol, ac_3_sol, ac_2_sol, ac_1_sol, ac_0_sol, \
           lb_x_sol, lb_y_sol, ub_x_sol, ub_y_sol, pde_sol, txy_ki, \
        #             txy_zero\


#             ,txy_kibx, txy_kiby, txy_stk5x, txy_stk5y


loss_u = 1
loss_pde_u = 1
loss_ic_u = 1
loss_ac_4_u = 1
loss_ac_3_u = 1
loss_ac_2_u = 1
loss_ac_1_u = 1
loss_ac_0_u = 1
loss_lb_x_u = 1
loss_lb_y_u = 1
loss_ub_x_u = 1
loss_ub_y_u = 1
loss_u_txy_ki = 1
loss_history_u = []
time_history_u = []
step_history_u = []

loss_ku = 1
loss_pde_ku = 1
loss_ic_ku = 1
loss_ac_4_ku = 1
loss_ac_3_ku = 1
loss_ac_2_ku = 1
loss_ac_1_ku = 1
loss_ac_0_ku = 1
loss_lb_x_ku = 1
loss_lb_y_ku = 1
loss_ub_x_ku = 1
loss_ub_y_ku = 1
loss_history_ku = []
time_history_ku = []
step_history_ku = []

tensor1_t = torch.FloatTensor([2.9]).reshape(1, 1)
tensor1_x = torch.FloatTensor([0.6]).reshape(1, 1)
tensor1_y = torch.FloatTensor([0.8]).reshape(1, 1)
tensor1_to_device = torch.cat((tensor1_t, tensor1_x, tensor1_y), 1).to(device)
tensor1_to_device.requires_grad = True
nn5 = NN5(tensor1_to_device)

# first order derivatives
nn5_1 = torch.autograd.grad(inputs=tensor1_to_device, outputs=nn5, grad_outputs=torch.ones_like(nn5), retain_graph=True,
                            create_graph=True)[0]
print('nn5_1 : {}'.format(nn5_1))
nn5_x = torch.matmul(nn5_1, torch.FloatTensor([0., 1., 0.]).to(device).reshape(3, 1))
nn5_y = torch.matmul(nn5_1, torch.FloatTensor([0., 0., 1.]).to(device).reshape(3, 1))
# print('u_t : {}'.format(nn5_1[:,0]))
print('u_x : {}'.format(nn5_x))
print('u_y : {}'.format(nn5_y))

# second order derivatives
nn5_xy = \
torch.autograd.grad(inputs=tensor1_to_device, outputs=nn5_x, grad_outputs=torch.ones_like(nn5_x), retain_graph=True,
                    create_graph=True)[0]
nn5_yx = \
torch.autograd.grad(inputs=tensor1_to_device, outputs=nn5_y, grad_outputs=torch.ones_like(nn5_y), retain_graph=True,
                    create_graph=True)[0]
# print('nn5_2 : {}'.format(nn5_2))
print('u_xy : {}'.format(nn5_xy[:, 2]))
print('u_yx : {}'.format(nn5_yx[:, 1]))

start_time = time.time()

epochs = 2000  # 200

from tqdm import tqdm

for epoch in tqdm(range(epochs)):
    if (loss_u <= 1e-5 and loss_ku <= 1e-5):
        break

    _ = 0
    x, y, txy, ic, ac_4, ac_3, ac_2, ac_1, ac_0, lb_x, lb_y, ub_x, ub_y, ic_sol, kic_sol, \
    ac_4_sol, ac_3_sol, ac_2_sol, ac_1_sol, ac_0_sol, \
    lb_x_sol, lb_y_sol, ub_x_sol, ub_y_sol, pde_sol, txy_ki \
        = generate_data_bspde_gbm_put_3param_ELS(T, L, r)

    # No KI ELS
    while (loss_u > 1e-5 or loss_ku > 1e-5) and _ < 500:

        _ += 1

        u = NN5(txy)
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

        u_ic = NN5(ic)
        u_ac_4 = NN5(ac_4)
        u_ac_3 = NN5(ac_3)
        u_ac_2 = NN5(ac_2)
        u_ac_1 = NN5(ac_1)
        u_ac_0 = NN5(ac_0)

        u_lb_x = NN5(lb_x)
        u_lb_y = NN5(lb_y)

        u_ub_x = NN5(ub_x)
        u_ub_y = NN5(ub_y)

        u_lb_x_x = torch.autograd.grad(inputs=lb_x, outputs=u_lb_x, grad_outputs=torch.ones_like(u_lb_x),
                                       retain_graph=True, create_graph=True)[0][:, 1].reshape(-1, 1)
        u_lb_x_xx = \
        torch.autograd.grad(inputs=lb_x, outputs=u_lb_x_x, grad_outputs=torch.ones_like(u_lb_x_x), retain_graph=True,
                            create_graph=True)[0][:, 1].reshape(-1, 1)

        u_lb_y_y = \
        torch.autograd.grad(inputs=lb_y, outputs=u_lb_y, grad_outputs=torch.ones_like(u_lb_y), retain_graph=True,
                            create_graph=True)[0][:, 2].reshape(-1, 1)
        u_lb_y_yy = \
        torch.autograd.grad(inputs=lb_y, outputs=u_lb_y_y, grad_outputs=torch.ones_like(u_lb_y_y), retain_graph=True,
                            create_graph=True)[0][:, 2].reshape(-1, 1)

        u_ub_x_x = \
        torch.autograd.grad(inputs=ub_x, outputs=u_ub_x, grad_outputs=torch.ones_like(u_ub_x), retain_graph=True,
                            create_graph=True)[0][:, 1].reshape(-1, 1)
        u_ub_x_xx = \
        torch.autograd.grad(inputs=ub_x, outputs=u_ub_x_x, grad_outputs=torch.ones_like(u_ub_x_x), retain_graph=True,
                            create_graph=True)[0][:, 1].reshape(-1, 1)

        u_ub_y_y = \
        torch.autograd.grad(inputs=ub_y, outputs=u_ub_y, grad_outputs=torch.ones_like(u_ub_y), retain_graph=True,
                            create_graph=True)[0][:, 2].reshape(-1, 1)
        u_ub_y_yy = \
        torch.autograd.grad(inputs=ub_y, outputs=u_ub_y_y, grad_outputs=torch.ones_like(u_ub_y_y), retain_graph=True,
                            create_graph=True)[0][:, 2].reshape(-1, 1)

        u_txy_ki = NN5(txy_ki)

        ku = NN6(txy)
        ku_1 = torch.autograd.grad(inputs=txy, outputs=ku, grad_outputs=torch.ones_like(ku), retain_graph=True,
                                   create_graph=True)[0]
        ku_t = ku_1[:, 0].reshape(-1, 1)
        ku_x = ku_1[:, 1].reshape(-1, 1)
        ku_y = ku_1[:, 2].reshape(-1, 1)
        ku_2 = torch.autograd.grad(inputs=txy, outputs=ku_x, grad_outputs=torch.ones_like(ku_x), retain_graph=True,
                                   create_graph=True)[0]
        ku_xx = ku_2[:, 1].reshape(-1, 1)
        ku_xy = ku_2[:, 2].reshape(-1, 1)
        ku_yy = torch.autograd.grad(inputs=txy, outputs=ku_y, grad_outputs=torch.ones_like(ku_y), retain_graph=True,
                                    create_graph=True)[0][:, 2].reshape(-1, 1)

        # 여기서부터 이어서 작업

        ku_ic = NN6(ic)
        ku_ac_4 = NN6(ac_4)
        ku_ac_3 = NN6(ac_3)
        ku_ac_2 = NN6(ac_2)
        ku_ac_1 = NN6(ac_1)
        ku_ac_0 = NN6(ac_0)

        ku_lb_x = NN6(lb_x)
        ku_lb_y = NN6(lb_y)

        ku_ub_x = NN6(ub_x)
        ku_ub_y = NN6(ub_y)

        ku_lb_x_x = torch.autograd.grad(inputs=lb_x, outputs=ku_lb_x,
                                        grad_outputs=torch.ones_like(ku_lb_x), retain_graph=True, create_graph=True)[0][
                    :, 1].reshape(-1, 1)
        ku_lb_x_xx = torch.autograd.grad(inputs=lb_x, outputs=ku_lb_x_x,
                                         grad_outputs=torch.ones_like(ku_lb_x_x), retain_graph=True, create_graph=True)[
                         0][:, 1].reshape(-1, 1)

        ku_lb_y_y = torch.autograd.grad(inputs=lb_y, outputs=ku_lb_y,
                                        grad_outputs=torch.ones_like(ku_lb_y), retain_graph=True, create_graph=True)[0][
                    :, 2].reshape(-1, 1)
        ku_lb_y_yy = torch.autograd.grad(inputs=lb_y, outputs=ku_lb_y_y,
                                         grad_outputs=torch.ones_like(ku_lb_y_y), retain_graph=True, create_graph=True)[
                         0][:, 2].reshape(-1, 1)

        ku_ub_x_x = torch.autograd.grad(inputs=ub_x, outputs=ku_ub_x,
                                        grad_outputs=torch.ones_like(ku_ub_x), retain_graph=True, create_graph=True)[0][
                    :, 1].reshape(-1, 1)
        ku_ub_x_xx = torch.autograd.grad(inputs=ub_x, outputs=ku_ub_x_x,
                                         grad_outputs=torch.ones_like(ku_ub_x_x), retain_graph=True, create_graph=True)[
                         0][:, 1].reshape(-1, 1)

        ku_ub_y_y = torch.autograd.grad(inputs=ub_y, outputs=ku_ub_y,
                                        grad_outputs=torch.ones_like(ku_ub_y), retain_graph=True, create_graph=True)[0][
                    :, 2].reshape(-1, 1)
        ku_ub_y_yy = torch.autograd.grad(inputs=ub_y, outputs=ku_ub_y_y,
                                         grad_outputs=torch.ones_like(ku_ub_y_y), retain_graph=True, create_graph=True)[
                         0][:, 2].reshape(-1, 1)

        ku_txy_ki = NN6(txy_ki)

        pde_u = u_t - 0.5 * LV2_x * x ** 2 * u_xx - 0.5 * LV2_y * y ** 2 * u_yy \
                - corr * LV_x * LV_y * x * y * u_xy \
                - (r - d_x) * x * u_x - (r - d_y) * y * u_y + r * u

        pde_ku = ku_t - 0.5 * LV2_x * x ** 2 * ku_xx - 0.5 * LV2_y * y ** 2 * ku_yy \
                 - corr * LV_x * LV_y * x * y * ku_xy \
                 - (r - d_x) * x * ku_x - (r - d_y) * y * ku_y + r * ku

        loss_pde_u = mse_cost_function(pde_u, pde_sol)
        loss_ic_u = mse_cost_function(u_ic, ic_sol)
        loss_ac_4_u = mse_cost_function(u_ac_4, ac_4_sol)
        loss_ac_3_u = mse_cost_function(u_ac_3, ac_3_sol)
        loss_ac_2_u = mse_cost_function(u_ac_2, ac_2_sol)
        loss_ac_1_u = mse_cost_function(u_ac_1, ac_1_sol)
        loss_ac_0_u = mse_cost_function(u_ac_0, ac_0_sol)

        #         loss_lb_x_u = mse_cost_function(u_lb_x, lb_x_sol)
        #         loss_lb_y_u = mse_cost_function(u_lb_y, lb_y_sol)
        loss_lb_x_u = mse_cost_function(u_lb_x_xx, lb_x_sol) + mse_cost_function(u_lb_x, lb_x_sol)
        loss_lb_y_u = mse_cost_function(u_lb_y_yy, lb_y_sol) + mse_cost_function(u_lb_y, lb_y_sol)
        loss_ub_x_u = mse_cost_function(u_ub_x_xx, ub_x_sol)
        loss_ub_y_u = mse_cost_function(u_ub_y_yy, ub_y_sol)

        #         loss_u_txy_ki = mse_cost_function(u_txy_ki, ku_txy_ki.clone().detach())
        loss_u_txy_ki = 0

        loss_pde_ku = mse_cost_function(pde_ku, pde_sol)
        loss_ic_ku = mse_cost_function(ku_ic, kic_sol)
        loss_ac_4_ku = mse_cost_function(ku_ac_4, ac_4_sol)
        loss_ac_3_ku = mse_cost_function(ku_ac_3, ac_3_sol)
        loss_ac_2_ku = mse_cost_function(ku_ac_2, ac_2_sol)
        loss_ac_1_ku = mse_cost_function(ku_ac_1, ac_1_sol)
        loss_ac_0_ku = mse_cost_function(ku_ac_0, ac_0_sol)

        #         loss_lb_x_ku = mse_cost_function(ku_lb_x, lb_x_sol)
        #         loss_lb_y_ku = mse_cost_function(ku_lb_y, lb_y_sol)
        loss_lb_x_ku = mse_cost_function(ku_lb_x_xx, lb_x_sol) + mse_cost_function(ku_lb_x, lb_x_sol)
        loss_lb_y_ku = mse_cost_function(ku_lb_y_yy, lb_y_sol) + mse_cost_function(ku_lb_y, lb_y_sol)
        loss_ub_x_ku = mse_cost_function(ku_ub_x_xx, ub_x_sol)
        loss_ub_y_ku = mse_cost_function(ku_ub_y_yy, ub_y_sol)

        loss_u = loss_pde_u + 1 * loss_ic_u \
                 + 1 * (loss_lb_x_u + loss_lb_y_u + loss_ub_x_u + loss_ub_y_u) \
                 + 1 * (loss_ac_4_u + loss_ac_3_u + loss_ac_2_u + loss_ac_1_u + loss_ac_0_u) \
                 + loss_u_txy_ki \
 \
        loss_ku = loss_pde_ku + 1 * loss_ic_ku \
                  + 1 * (loss_lb_x_ku + loss_lb_y_ku + loss_ub_x_ku + loss_ub_y_ku) \
                  + 1 * (loss_ac_4_ku + loss_ac_3_ku + loss_ac_2_ku + loss_ac_1_ku + loss_ac_0_ku) \
 \
        optimizer5.step(closure_u_Adam)
        optimizer6.step(closure_ku_Adam)

        #        Plotting Test sample (121 x 121) mesh
        with torch.autograd.no_grad():
            if _ % 500 == 0 and epoch % 1 == 0:
                print('step : {}, epoch : {}'.format(_, epoch))
                print('loss_u : {}\n' \
                      'loss_pde_u : {}\n' \
                      'loss_ic_u : {}, loss_ac_4_u : {}, loss_ac_3_u : {}\n' \
                      'loss_ac_2_u : {}, loss_ac_1_u : {}, loss_ac_0_u : {}\n' \
                      'loss_lb_x_u : {}, loss_lb_y_u : {}\n' \
                      'loss_ub_x_u : {}, loss_ub_y_u : {}\n' \
                      'loss_u_txy_ki : {}'.format(loss_u, loss_pde_u,
                                                  loss_ic_u, loss_ac_4_u, loss_ac_3_u, loss_ac_2_u, loss_ac_1_u,
                                                  loss_ac_0_u,
                                                  loss_lb_x_u, loss_lb_y_u, loss_ub_x_u, loss_ub_y_u, loss_u_txy_ki))

                print('time : {}'.format(time.time() - start_time))
                u_outcome = torch.tensor([T, 1.0, 1.0]).reshape((1, 3)).to(device)
                print('u price : {}'.format(NN5(u_outcome) * facevalue))

            if _ % 1 == 0 and epoch % 1 == 0:
                loss_history_u.append(loss_u.item())
                time_history_u.append(time.time() - start_time)

            if _ % 500 == 0 and epoch % 1 == 0:
                ONES = torch.ones((121 ** 2, 1))
                ZEROS = torch.zeros((121 ** 2, 1))
                xxx = torch.linspace(0., L, 121)  # .reshape(121,1)
                yyy = xxx
                XXX, YYY = torch.meshgrid(xxx, yyy)
                XXX_flat = XXX.flatten()[:, None]
                YYY_flat = YYY.flatten()[:, None]
                XY = torch.cat((XXX_flat, YYY_flat), axis=1)
                TXY0 = torch.cat((step[5] * ONES, XY), axis=1).to(device)
                TXY1 = torch.cat((T * ONES, XY), axis=1).to(device)
                TXY2 = torch.cat((step[5] * ONES, XY), axis=1).to(device)
                TXY3 = torch.cat((T * ONES, XY), axis=1).to(device)

                NO_KI_ELS_mat0 = facevalue * NN5(TXY0).reshape(121, 121)
                NO_KI_ELS_mat1 = facevalue * NN5(TXY1).reshape(121, 121)
                NO_KI_ELS_mat2 = facevalue * NN5(TXY2).reshape(121, 121)
                NO_KI_ELS_mat3 = facevalue * NN5(TXY3).reshape(121, 121)

                fig = plt.figure(figsize=(10, 10))
                ax0 = fig.add_subplot(221, projection='3d')
                ax1 = fig.add_subplot(222, projection='3d')
                ax2 = fig.add_subplot(223, projection='3d')
                ax3 = fig.add_subplot(224, projection='3d')

                surf0 = ax0.plot_surface(XXX, YYY, NO_KI_ELS_mat0.cpu().numpy(), cmap=cm.coolwarm)
                surf1 = ax1.plot_surface(XXX, YYY, NO_KI_ELS_mat1.cpu().numpy(), cmap=cm.coolwarm)
                surf2 = ax2.plot_surface(XXX, YYY, NO_KI_ELS_mat2.cpu().numpy(), cmap=cm.coolwarm)
                surf3 = ax3.plot_surface(XXX, YYY, NO_KI_ELS_mat3.cpu().numpy(), cmap=cm.coolwarm)

                ax0.view_init(20, -45)
                ax1.view_init(20, -45)
                ax2.view_init(20, -135)
                ax3.view_init(20, -135)

                ax0.set_zlim3d(0.0, 130.0)
                ax1.set_zlim3d(0.0, 130.0)
                ax2.set_zlim3d(0.0, 130.0)
                ax3.set_zlim3d(0.0, 130.0)

                ax0.set_title('NO KI ELS t=T')
                ax1.set_title('NO KI ELS t=0')
                ax2.set_title('NO KI ELS t=T')
                ax3.set_title('NO KI ELS t=0')

                # 그림 저장하기
                plt.savefig(time.strftime('Pictures/ELS/torch/NO_KI/ELS_price%Y-%m-%d-%H-%M-%S'), dpi=200)
                plt.show(block=False)
                plt.close()

            if _ % 500 == 0 and epoch % 1 == 0:
                print('step : {}, epoch : {}'.format(_, epoch))
                print('loss_ku : {}\n' \
                      'loss_pde_ku : {}\n' \
                      'loss_ic_ku : {}, loss_ac_4_ku : {}, loss_ac_3_ku : {}\n' \
                      'loss_ac_2_ku : {}, loss_ac_1_ku : {}, loss_ac_0_ku : {}\n' \
                      'loss_lb_x_ku : {}, loss_lb_y_ku : {}\n' \
                      'loss_ub_x_ku : {}, loss_ub_y_ku : {}'.format(loss_ku, loss_pde_ku,
                                                                    loss_ic_ku, loss_ac_4_ku, loss_ac_3_ku,
                                                                    loss_ac_2_ku, loss_ac_1_ku,
                                                                    loss_ac_0_ku, loss_lb_x_ku, loss_lb_y_ku,
                                                                    loss_ub_x_ku, loss_ub_y_ku))

                print('time : {}'.format(time.time() - start_time))
                ku_outcome = torch.tensor([T, 1.0 / L * L * 1, 1.0 / L * L * 1]).reshape((1, 3)).to(device)
                print('ku price : {}'.format(NN6(ku_outcome) * facevalue))

            if _ % 1 == 0 and epoch % 1 == 0:
                loss_history_ku.append(loss_ku.item())
                time_history_ku.append(time.time() - start_time)

            if _ % 500 == 0 and epoch % 1 == 0:
                ONES = torch.ones((121 ** 2, 1))
                ZEROS = torch.zeros((121 ** 2, 1))
                xxx = torch.linspace(0., L, 121)
                yyy = xxx
                XXX, YYY = torch.meshgrid(xxx, yyy)
                XXX_flat = XXX.flatten()[:, None]
                YYY_flat = YYY.flatten()[:, None]
                XY = torch.cat((XXX_flat, YYY_flat), axis=1)
                TXY0 = torch.cat((step[5] * ONES, XY), axis=1)
                TXY1 = torch.cat((T * ONES, XY), axis=1)
                TXY2 = torch.cat((step[5] * ONES, XY), axis=1)
                TXY3 = torch.cat((T * ONES, XY), axis=1)

                TXY0 = torch.FloatTensor(TXY0).to(device)
                TXY1 = torch.FloatTensor(TXY1).to(device)
                TXY2 = torch.FloatTensor(TXY2).to(device)
                TXY3 = torch.FloatTensor(TXY3).to(device)

                KI_ELS_mat0 = facevalue * NN6(TXY0).reshape(121, 121)
                KI_ELS_mat1 = facevalue * NN6(TXY1).reshape(121, 121)
                KI_ELS_mat2 = facevalue * NN6(TXY2).reshape(121, 121)
                KI_ELS_mat3 = facevalue * NN6(TXY3).reshape(121, 121)

                fig = plt.figure(figsize=(10, 10))
                ax0 = fig.add_subplot(221, projection='3d')
                ax1 = fig.add_subplot(222, projection='3d')
                ax2 = fig.add_subplot(223, projection='3d')
                ax3 = fig.add_subplot(224, projection='3d')

                surf0 = ax0.plot_surface(XXX, YYY, KI_ELS_mat0.cpu().numpy(), cmap=cm.coolwarm)
                surf1 = ax1.plot_surface(XXX, YYY, KI_ELS_mat1.cpu().numpy(), cmap=cm.coolwarm)
                surf2 = ax2.plot_surface(XXX, YYY, KI_ELS_mat2.cpu().numpy(), cmap=cm.coolwarm)
                surf3 = ax3.plot_surface(XXX, YYY, KI_ELS_mat3.cpu().numpy(), cmap=cm.coolwarm)

                ax0.view_init(20, -45)
                ax1.view_init(20, -45)
                ax2.view_init(20, -135)
                ax3.view_init(20, -135)

                ax0.set_zlim3d(0.0, 130.0)
                ax1.set_zlim3d(0.0, 130.0)
                ax2.set_zlim3d(0.0, 130.0)
                ax3.set_zlim3d(0.0, 130.0)

                ax0.set_title('KI ELS t=0')
                ax1.set_title('KI ELS t=T')

                plt.savefig(time.strftime('Pictures/ELS/torch/KI/ELS_price%Y-%m-%d-%H-%M-%S'), dpi=200)
                plt.show(block=False)
                plt.close()

    #         if _ == 10 and epoch % 1000 == 0 :
    #             date = time.strftime('%Y%m%d-%H%M%S')
    #             utils_bspde.save_weights_12(NN5, f'./save_weight/NN5_3param_12layers_20neurons_1to7_relu_softplus_{date}')
    #             utils_bspde.save_weights_12(NN6, f'./save_weight/NN6_3param_12layers_20neurons_1to7_relu_softplus_{date}')

    epoch_end = epoch

# date = time.strftime('%Y%m%d-%H%M%S')   
# utils_bspde.save_weights(NN5, f'./save_weight/NN5_2param_bisec_{date}')
# utils_bspde.save_weights(NN6, f'./save_weight/NN6_2param_bisec_{date}')

######
###### 학습이 끝난 후 loss
######
print('steps : {}'.format((epoch_end) * 500))

print('loss_u : {}\n' \
      'loss_pde_u : {}\n' \
      'loss_ic_u : {}, loss_ac_4_u : {}, loss_ac_3_u : {}\n' \
      'loss_ac_2_u : {}, loss_ac_1_u : {}, loss_ac_0_u : {}\n' \
      'loss_lb_x_u : {}, loss_lb_y_u : {}\n' \
      'loss_ub_x_u : {}, loss_ub_y_u : {}\n' \
      'loss_u_txy_ki : {}'.format(loss_u, loss_pde_u,
                                  loss_ic_u, loss_ac_4_u, loss_ac_3_u, loss_ac_2_u, loss_ac_1_u, loss_ac_0_u,
                                  loss_lb_x_u, loss_lb_y_u, loss_ub_x_u, loss_ub_y_u, loss_u_txy_ki))

print('loss_ku : {}\n' \
      'loss_pde_ku : {}\n' \
      'loss_ic_ku : {}, loss_ac_4_ku : {}, loss_ac_3_ku : {}\n' \
      'loss_ac_2_ku : {}, loss_ac_1_ku : {}, loss_ac_0_ku : {}\n' \
      'loss_lb_x_ku : {}, loss_lb_y_ku : {}\n' \
      'loss_ub_x_ku : {}, loss_ub_y_ku : {}'.format(loss_ku, loss_pde_ku,
                                                    loss_ic_ku, loss_ac_4_ku, loss_ac_3_ku, loss_ac_2_ku, loss_ac_1_ku,
                                                    loss_ac_0_ku,
                                                    loss_lb_x_ku, loss_lb_y_ku, loss_ub_x_ku, loss_ub_y_ku))

x_len_u = torch.arange(len(loss_history_u))
x_len_ku = torch.arange(len(loss_history_ku))

plt.subplot(211)
plt.plot(x_len_u, loss_history_u, marker='.', c='blue', label="Training Loss for u")
plt.legend(loc='upper right')
plt.grid()
plt.xlabel('Iteration')
plt.ylabel('loss')

plt.subplot(212)
plt.plot(x_len_ku, loss_history_ku, marker='.', c='blue', label="Training Loss for ku")
plt.legend(loc='upper right')
plt.grid()
plt.xlabel('Iteration')
plt.ylabel('loss')

plt.show()

print('loss_u : {}\n' \
      'loss_pde_u : {}\n' \
      'loss_ic_u : {}, loss_ac_4_u : {}, loss_ac_3_u : {}\n' \
      'loss_ac_2_u : {}, loss_ac_1_u : {}, loss_ac_0_u : {}\n' \
      'loss_lb_x_u : {}, loss_lb_y_u : {}\n' \
      'loss_ub_x_u : {}, loss_ub_y_u : {}\n' \
      'loss_u_txy_ki : {}'.format(loss_u, loss_pde_u,
                                  loss_ic_u, loss_ac_4_u, loss_ac_3_u, loss_ac_2_u, loss_ac_1_u, loss_ac_0_u,
                                  loss_lb_x_u, loss_lb_y_u, loss_ub_x_u, loss_ub_y_u, loss_u_txy_ki))

print('loss_ku : {}\n' \
      'loss_pde_ku : {}\n' \
      'loss_ic_ku : {}, loss_ac_4_ku : {}, loss_ac_3_ku : {}\n' \
      'loss_ac_2_ku : {}, loss_ac_1_ku : {}, loss_ac_0_ku : {}\n' \
      'loss_lb_x_ku : {}, loss_lb_y_ku : {}\n' \
      'loss_ub_x_ku : {}, loss_ub_y_ku : {}'.format(loss_ku, loss_pde_ku,
                                                    loss_ic_ku, loss_ac_4_ku, loss_ac_3_ku, loss_ac_2_ku, loss_ac_1_ku,
                                                    loss_ac_0_ku,
                                                    loss_lb_x_ku, loss_lb_y_ku, loss_ub_x_ku, loss_ub_y_ku))


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


# loss_u
x_len_u = torch.arange(len(loss_history_u))

plt.plot(x_len_u, torch.log10(torch.tensor(loss_history_u)), c='blue', label="Training Loss for u")
# plt.grid()
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.yticks(torch.tensor([torch.log10(1e-2), torch.log10(1e-1), torch.log10(1e-0)]),
           labels=['$10^{-2}$', '$10^{-1}$', '$1$'])
plt.savefig(time.strftime('Pictures/ELS/torch/Training_Loss_u%Y-%m-%d-%H-%M-%S'), dpi=200)

# loss_ku
x_len_ku = torch.arange(len(loss_history_ku))

plt.plot(x_len_ku, torch.log10(torch.tensor(loss_history_ku)), c='blue', label="Training Loss for ku")
# plt.grid()
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.yticks(torch.tensor([torch.log10(1e-2), torch.log10(1e-1), torch.log10(1e-0)]),
           labels=['$10^{-2}$', '$10^{-1}$', '$1$'])
plt.savefig(time.strftime('Pictures/ELS/torch/Training_Loss_ku%Y-%m-%d-%H-%M-%S'), dpi=200)

# u_ic
with torch.autograd.no_grad():
    ONES = torch.ones((121 ** 2, 1))
    ZEROS = torch.zeros((121 ** 2, 1))
    xxx = torch.linspace(0., L, 121).reshape(121, 1) / L * L * 1
    yyy = xxx
    XXX, YYY = torch.meshgrid(xxx, yyy)
    XXX_flat = XXX.flatten()[:, None]
    YYY_flat = YYY.flatten()[:, None]
    XY = torch.cat((XXX_flat, YYY_flat), axis=1)
    TXY = torch.cat((step[5] * ONES, XY), axis=1)
    TXY = torch.FloatTensor(TXY).to(device)

    NO_KI_ELS_mat = facevalue * NN5(TXY).reshape(121, 121)
    fig = plt.figure(figsize=(10, 10))

    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(XXX, YYY, NO_KI_ELS_mat.cpu().numpy(), cmap=cm.coolwarm)

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
    plt.savefig(time.strftime('Pictures/ELS/torch/PINN_u_maturity%Y-%m-%d-%H-%M-%S'), dpi=200)

# u
with torch.autograd.no_grad():
    ONES = torch.ones((121 ** 2, 1))
    ZEROS = torch.zeros((121 ** 2, 1))
    xxx = torch.linspace(0., L, 121).reshape(121, 1)
    yyy = xxx
    XXX, YYY = torch.meshgrid(xxx, yyy)
    XXX_flat = XXX.flatten()[:, None]
    YYY_flat = YYY.flatten()[:, None]
    XY = torch.cat((XXX_flat, YYY_flat), axis=1)
    TXY = torch.cat((T * ONES, XY), axis=1)
    TXY = torch.FloatTensor(TXY).to(device)

    NO_KI_ELS_mat = facevalue * NN5(TXY).reshape(121, 121)
    fig = plt.figure(figsize=(10, 10))

    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(XXX, YYY, NO_KI_ELS_mat.cpu().numpy(), cmap=cm.coolwarm)

    ax.view_init(20, -135)
    ax.set_zlim3d(0.0, 130.0)
    ax.set_xlabel('S1', fontsize=12)
    ax.set_ylabel('S2', fontsize=12)

    ax.set_xticks(torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]))
    ax.set_xticklabels(['$0$', '$50$', '$100$', '$150$', '$200$', '$250$', '$300$'])
    ax.set_yticks(torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]))
    ax.set_yticklabels(['$0$', '$50$', '$100$', '$150$', '$200$', '$250$', '$300$'])
    ax.set_title('ELS Price')
    #     fig.savefig('PINN_u.png')
    plt.savefig(time.strftime('Pictures/ELS/torch/PINN_u%Y-%m-%d-%H-%M-%S'), dpi=200)

# ku_ic
with torch.autograd.no_grad():
    ONES = torch.ones((121 ** 2, 1))
    ZEROS = torch.zeros((121 ** 2, 1))
    xxx = torch.linspace(0., L, 121).reshape(121, 1) / L * L * 1
    yyy = xxx
    XXX, YYY = torch.meshgrid(xxx, yyy)
    XXX_flat = XXX.flatten()[:, None]
    YYY_flat = YYY.flatten()[:, None]
    XY = torch.cat((XXX_flat, YYY_flat), axis=1)
    TXY = torch.cat((step[5] * ONES, XY), axis=1)
    TXY = torch.FloatTensor(TXY).to(device)

    KI_ELS_mat = facevalue * NN6(TXY).reshape(121, 121)
    fig = plt.figure(figsize=(10, 10))

    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(XXX, YYY, KI_ELS_mat.cpu().numpy(), cmap=cm.coolwarm)

    ax.view_init(20, -135)
    ax.set_zlim3d(0.0, 130.0)
    ax.set_xlabel('S1', fontsize=12)
    ax.set_ylabel('S2', fontsize=12)

    ax.set_xticks(torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]))
    ax.set_xticklabels(['$0$', '$50$', '$100$', '$150$', '$200$', '$250$', '$300$'])
    ax.set_yticks(torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]))
    ax.set_yticklabels(['$0$', '$50$', '$100$', '$150$', '$200$', '$250$', '$300$'])
    ax.set_title('ELS Price')
    #     fig.savefig('PINN_ku_maturity.png')
    plt.savefig(time.strftime('Pictures/ELS/torch/PINN_ku_maturity%Y-%m-%d-%H-%M-%S'), dpi=200)

# ku
with torch.autograd.no_grad():
    ONES = torch.ones((121 ** 2, 1))
    ZEROS = torch.zeros((121 ** 2, 1))
    xxx = torch.linspace(0., L, 121).reshape(121, 1)
    yyy = xxx
    XXX, YYY = torch.meshgrid(xxx, yyy)
    XXX_flat = XXX.flatten()[:, None]
    YYY_flat = YYY.flatten()[:, None]
    XY = torch.cat((XXX_flat, YYY_flat), axis=1)
    TXY = torch.cat((T * ONES, XY), axis=1)
    TXY = torch.FloatTensor(TXY).to(device)

    KI_ELS_mat = facevalue * NN6(TXY).reshape(121, 121)
    fig = plt.figure(figsize=(10, 10))

    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(XXX, YYY, KI_ELS_mat.cpu().numpy(), cmap=cm.coolwarm)

    ax.view_init(20, -135)
    ax.set_zlim3d(0.0, 130.0)
    ax.set_xlabel('S1', fontsize=12)
    ax.set_ylabel('S2', fontsize=12)

    ax.set_xticks(torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]))
    ax.set_xticklabels(['$0$', '$50$', '$100$', '$150$', '$200$', '$250$', '$300$'])
    ax.set_yticks(torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]))
    ax.set_yticklabels(['$0$', '$50$', '$100$', '$150$', '$200$', '$250$', '$300$'])
    ax.set_title('ELS Price')
    #     fig.savefig('PINN_ku.png')
    plt.savefig(time.strftime('Pictures/ELS/torch/ELS_Price%Y-%m-%d-%H-%M-%S'), dpi=200)

with torch.autograd.no_grad():
    u_outcome = torch.tensor([T, 1.0 / L * L * 1, 1.0 / L * L * 1]).reshape((1, 3))
    u_outcome = torch.FloatTensor(u_outcome).to(device)
    print('u price : {}'.format(NN5(u_outcome) * facevalue))

    ku_outcome = torch.tensor([T, 1.0 / L * L * 1, 1.0 / L * L * 1]).reshape((1, 3))
    ku_outcome = torch.FloatTensor(ku_outcome).to(device)
    print('ku price : {}'.format(NN6(ku_outcome) * facevalue))

# with torch.autograd.no_grad():
ONES = torch.ones((121 ** 2, 1))
ZEROS = torch.zeros((121 ** 2, 1))
xxx = torch.linspace(0., L, 121).reshape(121, 1) / L * L * 1
yyy = xxx
XXX, YYY = torch.meshgrid(xxx, yyy)
XXX_flat = XXX.flatten()[:, None]
YYY_flat = YYY.flatten()[:, None]
XY = torch.cat((XXX_flat, YYY_flat), axis=1)
TXY = torch.cat((T * ONES, XY), axis=1)
TXY = torch.FloatTensor(TXY).to(device)
TXY.requires_grad = True

uu = NN5(TXY)
uu_1 = torch.autograd.grad(inputs=TXY, outputs=uu, grad_outputs=torch.ones_like(uu),
                           retain_graph=True, create_graph=True)[0]
uu_t = uu_1[:, 0].reshape(-1, 1)
uu_x = uu_1[:, 1].reshape(-1, 1)
uu_y = uu_1[:, 2].reshape(-1, 1)
uu_2 = torch.autograd.grad(inputs=TXY, outputs=uu_x, grad_outputs=torch.ones_like(uu_x),
                           retain_graph=True, create_graph=True)[0]
uu_xx = uu_2[:, 1].reshape(-1, 1)
uu_xy = uu_2[:, 2].reshape(-1, 1)
uu_yy = torch.autograd.grad(inputs=TXY, outputs=uu_y, grad_outputs=torch.ones_like(uu_y),
                            retain_graph=True, create_graph=True)[0][:, 2].reshape(-1, 1)

delta_x_mat = uu_x.reshape(121, 121)
delta_y_mat = uu_y.reshape(121, 121)
gamma_x_mat = uu_xx.reshape(121, 121)
gamma_y_mat = uu_yy.reshape(121, 121)
crossgamma_xy_mat = uu_xy.reshape(121, 121)
theta_mat = uu_t.reshape(121, 121)

# u delta_x
with torch.autograd.no_grad():
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(XXX, YYY, delta_x_mat.cpu().numpy(), cmap=cm.coolwarm)

    ax.view_init(20, -135)
    ax.set_xlabel('S1', fontsize=12)
    ax.set_ylabel('S2', fontsize=12)

    ax.set_xticks(torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]))
    ax.set_xticklabels(['$0$', '$50$', '$100$', '$150$', '$200$', '$250$', '$300$'])
    ax.set_yticks(torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]))
    ax.set_yticklabels(['$0$', '$50$', '$100$', '$150$', '$200$', '$250$', '$300$'])
    ax.set_title('Delta 1')
    #     fig.savefig('PINN_u_delta_1.png')
    plt.savefig(time.strftime('Pictures/ELS/torch/PINN_u_delta_1%Y-%m-%d-%H-%M-%S'), dpi=200)

# u delta_y
with torch.autograd.no_grad():
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(XXX, YYY, delta_y_mat.cpu().numpy(), cmap=cm.coolwarm)

    ax.view_init(20, -135)
    ax.set_xlabel('S1', fontsize=12)
    ax.set_ylabel('S2', fontsize=12)

    ax.set_xticks(torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]))
    ax.set_xticklabels(['$0$', '$50$', '$100$', '$150$', '$200$', '$250$', '$300$'])
    ax.set_yticks(torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]))
    ax.set_yticklabels(['$0$', '$50$', '$100$', '$150$', '$200$', '$250$', '$300$'])
    ax.set_title('Delta 2')
    #     fig.savefig('PINN_u_delta_2.png')
    plt.savefig(time.strftime('Pictures/ELS/torch/PINN_u_delta_2_%Y-%m-%d-%H-%M-%S'), dpi=200)

# u gamma_x
with torch.autograd.no_grad():
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(XXX, YYY, gamma_x_mat.cpu().numpy(), cmap=cm.coolwarm)

    ax.view_init(20, -135)
    ax.set_xlabel('S1', fontsize=12)
    ax.set_ylabel('S2', fontsize=12)

    ax.set_xticks(torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]))
    ax.set_xticklabels(['$0$', '$50$', '$100$', '$150$', '$200$', '$250$', '$300$'])
    ax.set_yticks(torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]))
    ax.set_yticklabels(['$0$', '$50$', '$100$', '$150$', '$200$', '$250$', '$300$'])
    ax.set_title('Gamma 1')
    #     fig.savefig('PINN_u_gamma_1.png')
    plt.savefig(time.strftime('Pictures/ELS/torch/PINN_u_gamma_1_%Y-%m-%d-%H-%M-%S'), dpi=200)

# u gamma_y
with torch.autograd.no_grad():
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(XXX, YYY, gamma_y_mat.cpu().numpy(), cmap=cm.coolwarm)

    ax.view_init(20, -135)
    ax.set_xlabel('S1', fontsize=12)
    ax.set_ylabel('S2', fontsize=12)

    ax.set_xticks(torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]))
    ax.set_xticklabels(['$0$', '$50$', '$100$', '$150$', '$200$', '$250$', '$300$'])
    ax.set_yticks(torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]))
    ax.set_yticklabels(['$0$', '$50$', '$100$', '$150$', '$200$', '$250$', '$300$'])
    ax.set_title('Gamma 2')
    #     fig.savefig('PINN_u_gamma_2.png')
    plt.savefig(time.strftime('Pictures/ELS/torch/PINN_u_gamma_2_%Y-%m-%d-%H-%M-%S'), dpi=200)

# u crossgamma
# u gamma_y
with torch.autograd.no_grad():
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(XXX, YYY, crossgamma_xy_mat.cpu().numpy(), cmap=cm.coolwarm)

    ax.view_init(20, -135)
    ax.set_xlabel('S1', fontsize=12)
    ax.set_ylabel('S2', fontsize=12)

    ax.set_xticks(torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]))
    ax.set_xticklabels(['$0$', '$50$', '$100$', '$150$', '$200$', '$250$', '$300$'])
    ax.set_yticks(torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]))
    ax.set_yticklabels(['$0$', '$50$', '$100$', '$150$', '$200$', '$250$', '$300$'])
    ax.set_title('Cross gamma')
    #     fig.savefig('PINN_u_crossgamma.png')
    plt.savefig(time.strftime('Pictures/ELS/torch/PINN_u_crossgamma%Y-%m-%d-%H-%M-%S'), dpi=200)

# u theta
with torch.autograd.no_grad():
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(XXX, YYY, -theta_mat.cpu().numpy(), cmap=cm.coolwarm)

    ax.view_init(20, -135)
    ax.set_xlabel('S1', fontsize=12)
    ax.set_ylabel('S2', fontsize=12)

    ax.set_xticks(torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]))
    ax.set_xticklabels(['$0$', '$50$', '$100$', '$150$', '$200$', '$250$', '$300$'])
    ax.set_yticks(torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]))
    ax.set_yticklabels(['$0$', '$50$', '$100$', '$150$', '$200$', '$250$', '$300$'])
    ax.set_title('Theta')
    #     fig.savefig('PINN_u_theta.png')
    plt.savefig(time.strftime('Pictures/ELS/torch/PINN_u_theta%Y-%m-%d-%H-%M-%S'), dpi=200)
