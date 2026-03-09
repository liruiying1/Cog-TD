from syn1_copulamodel import JointModel
import torch
import torch.nn as nn
import torch.distributions as dist
from torch import optim
from scipy.io import loadmat, savemat
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from VCA import vca
from loss_copula import SAD,NonZeroClipper,SparseKLloss,compute_rmse,SumToOneLoss
import time

import scipy.io as sio

L = 224
P = 3
nr1 = 50
nc1 = 50
T = 6

data1 = sio.loadmat('/home/home_node6_1/lry/MTHU/20241115resnet/result/single/syn1/1syn180_1.mat')
data2 = sio.loadmat('/home/home_node6_1/lry/MTHU/20241115resnet/result/single/syn1/2syn180_1.mat')
data3 = sio.loadmat('/home/home_node6_1/lry/MTHU/20241115resnet/result/single/syn1/3syn180_1.mat')
data4 = sio.loadmat('/home/home_node6_1/lry/MTHU/20241115resnet/result/single/syn1/4syn180_1.mat')
data5 = sio.loadmat('/home/home_node6_1/lry/MTHU/20241115resnet/result/single/syn1/5syn180_1.mat')
data6 = sio.loadmat('/home/home_node6_1/lry/MTHU/20241115resnet/result/single/syn1/6syn180_1.mat')

abu0 = torch.from_numpy(data1['abu_est']).float()
abu1 = torch.from_numpy(data2['abu_est']).float()
abu2 = torch.from_numpy(data3['abu_est']).float()
abu3 = torch.from_numpy(data4['abu_est']).float()
abu4 = torch.from_numpy(data5['abu_est']).float()
abu5 = torch.from_numpy(data6['abu_est']).float()

abu = torch.stack([abu0,abu1, abu2, abu3, abu4, abu5])
abu_true = abu
abu = abu.reshape(T,P,nr1*nc1)

# fix random seeds for reproducibility
SEED = 205
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def compute_rmse(x_true, x_pre):
    img_w, img_h, img_c = x_true.shape
    return np.sqrt( ((x_true-x_pre)**2).sum()/(img_w*img_h*img_c) )

def min_max(x):
    # x: numpy array
    x = (x - torch.min(x)) / (torch.max(x) - torch.min(x))
    return x

def compute_f_integrated(f_hat):
    """
    使用梯形法则计算 PDF 的积分。
    参数:
    - f_hat: 预测的 PDF 值，形状为 [batch_size, num_points]
    - x_points: 积分点，形状为 [num_points]    
    返回:
    - f_integrated: CDF 值，形状为 [batch_size, num_points]
    """
    num_points = 7500
    x_points = torch.linspace(0, 1, num_points)
    # 计算每个小段的宽度
    delta_x = x_points[1:] - x_points[:-1]

    cdf = torch.zeros_like(f_hat)

    for i in range(f_hat.shape[0]):
        cumulative_sum = 0.0
        for j in range(f_hat.shape[1]):
            cumulative_sum += f_hat[i, j] * delta_x[0]
            cdf[i, j] = cumulative_sum
    
    return cdf

def main():
    iter_rec, loss_rec = list(), list()

    # get data and pass it through inference net
    model = JointModel()
    print(model)
    
    model.apply(model.weights_init)
    #model.to(device)

    optim = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=3e-5, betas=(0.9, 0.999), eps=1e-8)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=1, gamma=0.92)
    time_start = time.time()

    epoch = 100
    for iter in range(epoch):
        model.train()
        m_CDF_c, m_PDF_c, c_CDF_c, c_PDF_c, j_PDF_c = model(abu.float())

        zero_tensor = torch.zeros(T,P,nr1*nc1) 
        one_tensor = torch.ones(T,P,nr1*nc1)

        bound_zero, _, copula_zero, _, _ = model(zero_tensor.float())
        bound_one, _, copula_one, _, _ = model(one_tensor.float())

        j_f_integrated = compute_f_integrated(j_PDF_c.cpu())
        print('j_f_integrated',j_f_integrated.shape)

        # 创建边界条件张量
        F_boundary = np.zeros((nc1*nr1, 2))
        F_boundary[:, 0] = 0.0  # 最小值处的 CDF 值为 0
        F_boundary[:, 1] = 1.0  # 最大值处的 CDF 值为 1

        # 转换为 PyTorch 张量
        F_boundary = np.array(F_boundary)
        F_boundary = torch.tensor(F_boundary, dtype=torch.float32)

        # 计算累积概率
        F_observeds = []
        for i in range(T):
            sorted_data = np.sort(abu.reshape(T,P*nr1*nc1)[i,:])
            F_observed = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
            F_observeds.append(F_observed)       
        # 转换为 PyTorch 张量
        F_observeds = torch.tensor(F_observeds, dtype=torch.float32)
        print('F_observeds',F_observeds.shape)
                
        # 1. 对数损失
        m_log_loss = -torch.mean(torch.log(m_PDF_c + 1e-8))
        # 2. 非负性惩罚
        m_relu_loss = torch.mean(torch.relu(-m_PDF_c))
        # 3. 积分约束
        m_integral_loss = torch.abs(torch.sum(m_CDF_c) - 1.0)
        # 4. 边界条件约束
        m_boundary_loss = torch.sum(bound_zero)+torch.sum(torch.abs(1 - bound_one))

        m_loss = m_log_loss+m_relu_loss+m_integral_loss+m_boundary_loss

        # 1. 对数损失
        c_log_loss = -torch.mean(torch.log(j_PDF_c + 1e-8))  # 添加一个小的值以避免log(0)
        # 2. 非负性惩罚
        c_relu_loss = torch.mean(torch.relu(-j_PDF_c))
        # 3. 积分约束
        c_integral_loss = torch.abs(torch.sum(j_f_integrated) - 1.0)
        # 4. 边界条件约束
        c_boundary_loss = torch.sum(copula_zero)+torch.sum(torch.abs(1 - copula_one))

        # 5. 观测值约束
        observed_loss = torch.mean(torch.abs(F_observeds - j_f_integrated))

        c_loss = c_log_loss+c_relu_loss+c_integral_loss+c_boundary_loss+observed_loss


        total_loss = m_loss+c_loss
        
        optim.zero_grad()
        total_loss.backward()
        optim.step()
        torch.cuda.empty_cache()
        scheduler.step()
        if iter % 10 == 0:
            print('Epoch:', iter, '| loss: %.4f' % total_loss.item(),'| m_loss: %.4f' % m_loss.item(),'| c_loss: %.4f' % c_loss.item())
            
        iter_rec.append(iter)
    
    time_end = time.time()
    total_time = time_end - time_start
    print('The training lasts for {} seconds'.format(total_time))
        
    print('-------------------START EVAL---------------------')
    model.eval()
    m_CDF_c, m_PDF_c, c_CDF_c, c_PDF_c, j_PDF_c = model(abu.float())

    return m_CDF_c, m_PDF_c, c_CDF_c, c_PDF_c, j_PDF_c
    
if __name__ == '__main__':
    t_start = time.time() # start timer
    m_CDF_c, m_PDF_c, c_CDF_c, c_PDF_c, j_PDF_c = main()
    t_elapsed = time.time() - t_start # measure elapsed time

    
    matpath = '/home/home_node6_1/lry/MTHU/202503copula/code_Cog-TD/syn1_copula.mat'
    sio.savemat(matpath, \
                 {
                  'marginal_CDF': m_CDF_c.cpu().detach().numpy(),
                  "marginal_PDF": m_PDF_c.cpu().detach().numpy(),
                  'copula_CDF': c_CDF_c.cpu().detach().numpy(),
                  "copula_PDF": c_PDF_c.cpu().detach().numpy(),
                  'joint_PDF' : j_PDF_c.cpu().detach().numpy()})
   