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
from Data_Gen import generate_data_bspde_gbm_put_3param_ELS
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
N_ic = 5000  # 500  #초기값 점들 수
N_ac = [1000, 1000, 1000, 1000, 1000]  # N_ac_4,3,2,1,0
N_b = [1000, 1000, 1000, 1000]  # lb_x,lb_y,ub_x,ub_y

Ns = (N_coll, N_ic, N_ac, N_b)
N_sam = N_coll + N_ic + np.sum(N_ac) + np.sum(N_b)

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

epochs = 400  # 200

for epoch in tqdm(range(epochs)):
    if (loss_u <= 1e-5 and loss_ku <= 1e-5):
        break

    _ = 0
    x, y, txy, ic, ac_4, ac_3, ac_2, ac_1, ac_0, lb_x, lb_y, ub_x, ub_y, ic_sol, kic_sol, \
    ac_4_sol, ac_3_sol, ac_2_sol, ac_1_sol, ac_0_sol, \
    lb_x_sol, lb_y_sol, ub_x_sol, ub_y_sol, pde_sol, txy_ki \
        = generate_data_bspde_gbm_put_3param_ELS(T, L, r, Ns, kib, coupon, step, Strike)

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
        u_txy_ki = NN5(txy_ki)

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

        ku_txy_ki = NN6(txy_ki)
        ku_txy_ki_cloned = ku_txy_ki.clone().detach()

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
        # loss_ub_x_u = mse_cost_function(u_ub_x_xx, ub_x_sol)
        # loss_ub_y_u = mse_cost_function(u_ub_y_yy, ub_y_sol)

        loss_ub_x_u = mse_cost_function(u_ub_x_x, ub_x_sol)
        loss_ub_y_u = mse_cost_function(u_ub_y_y, ub_y_sol)


        # print(ku_txy_ki_cloned.requires_grad)
        loss_u_txy_ki = mse_cost_function(u_txy_ki, ku_txy_ki_cloned)
        # loss_u_txy_ki = mse_cost_function(u_txy_ki, torch.zeros_like(u_txy_ki))
        # loss_u_txy_ki = 0

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
        # loss_ub_x_ku = mse_cost_function(ku_ub_x_xx, ub_x_sol)
        # loss_ub_y_ku = mse_cost_function(ku_ub_y_yy, ub_y_sol)

        loss_ub_x_ku = mse_cost_function(ku_ub_x_x, ub_x_sol)
        loss_ub_y_ku = mse_cost_function(ku_ub_y_y, ub_y_sol)

        loss_u = loss_pde_u + 1 * loss_ic_u \
                 + 1 * (loss_lb_x_u + loss_lb_y_u + loss_ub_x_u + loss_ub_y_u) \
                 + 1 * (loss_ac_4_u + loss_ac_3_u + loss_ac_2_u + loss_ac_1_u + loss_ac_0_u) \
                 + loss_u_txy_ki

        loss_ku = loss_pde_ku + 1 * loss_ic_ku \
                  + 1 * (loss_lb_x_ku + loss_lb_y_ku + loss_ub_x_ku + loss_ub_y_ku) \
                  + 1 * (loss_ac_4_ku + loss_ac_3_ku + loss_ac_2_ku + loss_ac_1_ku + loss_ac_0_ku)
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

    epoch_end = epoch
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
plot_loss(loss_history_u, "loss")
plot_ELS(step[5], L, 121, device, facevalue, NN5, "PINN_u_maturity")
plot_ELS(T, L, 121, device, facevalue, NN6, "PINN_ku_maturity")
plot_ELS(0, L, 121, device, facevalue, NN5, "PINN_u_maturity0")
plot_ELS(0, L, 121, device, facevalue, NN6, "PINN_ku_maturity0")