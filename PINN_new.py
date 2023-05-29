#!/usr/bin/env python
# coding: utf-8
# torchPINN20230328을 완전 torch로 바꾸자 2023.03.30 힘드네

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

import torch
import torch.nn as nn
from tqdm import tqdm

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib as mat
from pylab import cm
from Data_Gen import generate_data_bspde_gbm_put_3param_ELS, generate_data_step, \
    generate_sol_step, output_PINN, Loss_PINN
from Plot import plot_ELS, plot_loss

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

N_coll = 10000  # 0 #100000 #10000 # Number of collocation points
N_ic = 1000  # 500  #초기값 점들 수
# N_ac = [1000, 1000, 1000, 1000, 1000]  # N_ac_4,3,2,1,0
# N_ac = [1000, 1000, 1000, 1000, 1000]  # N_ac_4,3,2,1,0
N_b = [500, 500, 500, 500]  # lb_x,lb_y,ub_x,ub_y

Ns = (N_coll, N_ic, N_b)
N_sam = N_coll + N_ic + np.sum(N_b)

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

# ELS price for No touch kib
KNN = []
for i in range(6):
    KNN.append(PINN_BS_2D().to(device))  # ELS price for touch kib
mse_cost_function = torch.nn.MSELoss().to(device)  # Mean squared error

# Adam optimizer: lr, beta1, beta2, epsilon default setting
optimizers = []
for nn in KNN:
    optimizers.append(torch.optim.Adam(nn.parameters()))


def closure_nn_Adam(optimizer, loss):  # u는 knock in 안 했을 때 가격
    if torch.is_grad_enabled():
        optimizer.zero_grad()
    if loss.requires_grad:
        loss.backward()
    return loss


loss_u = 1
loss_history_u = []
time_history_u = []
step_history_u = []

start_time = time.time()

epochs = 200  # 200
i = 5
while i >= 0:
    nn = KNN[i]
    print(i)
    if i == 5:
        init = True
        nn_pre = None
        barrier = kib
    else:
        init = False
        barrier = Strike[i]
        nn_pre = KNN[i + 1]
        nn_pre.eval()
    for epoch in tqdm(range(epochs)):
        if loss_u <= 1e-5:
            break
        _ = 0

        if i == 0:
            txy, ic, lb_x, lb_y, ub_x, ub_y, x, y = \
                generate_data_step(T, L, Ns, xbound=None, ybound=None, tbound=(step[i], T))
        else:
            txy, ic, lb_x, lb_y, ub_x, ub_y, x, y = \
                generate_data_step(T, L, Ns, xbound=None, ybound=None, tbound=(step[i], step[i - 1]))
        while loss_u > 1e-5 and _ < 500:
            _ += 1
            us = output_PINN(nn, txy, ic, lb_x, lb_y, ub_x, ub_y)

            # loss_u = Loss_PINN(u, u1, u2, u_ic, ub, ub1, ub2, x, y,coupon[5], kib, strike=Strike[i], init=init, nn_pre=nn_pre)
            loss_u = Loss_PINN(us, Ns, ic, x, y, mse_cost_function, coupon[i], barrier, init=init, nn_pre=nn_pre)
            # print(ku_txy_ki_cloned.requires_grad)
            closure_nn_Adam(optimizers[i], loss_u)
            optimizers[i].step()

            #        Plotting Test sample (121 x 121) mesh
            with torch.autograd.no_grad():
                if _ % 500 == 0 and epoch % 1 == 0:
                    print('step : {}, epoch : {}'.format(_, epoch))
                    print('loss_u : {}'.format(loss_u))
                    print('time : {}'.format(time.time() - start_time))

                if _ % 1 == 0 and epoch % 1 == 0:
                    loss_history_u.append(loss_u.item())
                    time_history_u.append(time.time() - start_time)
        epoch_end = epoch
    plot_ELS(step[i], L, 121, device, facevalue, nn, "PINN_u_maturity_init" + str(i))
    if i is not 0:
        plot_ELS(step[i - 1], L, 121, device, facevalue, nn, "PINN_u_maturity" + str(i))
    else:
        plot_ELS(T, L, 121, device, facevalue, nn, "PINN_u_maturity" + str(i))
    # plot_ELS(step[i - 1], L, 121, device, facevalue, nn, "PINN_ku_maturity")

    i = i - 1
for idx, nn in enumerate(KNN):
    torch.save(nn, f'./Model/KNN{idx}_0529.pth')
