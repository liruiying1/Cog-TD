import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec, colors
from datetime import datetime
from sklearn.manifold import TSNE
from scipy.io import loadmat, savemat
import argparse
from VCA import *
import scipy.io as sio
import time

# 数据加载
L = 224
P = 3
nr1 = 50
nc1 = 50
T = 6

data1 = sio.loadmat('.../single/syn1/t1.mat')
data2 = sio.loadmat('.../single/syn1/t2.mat')
data3 = sio.loadmat('.../single/syn1/t3.mat')
data4 = sio.loadmat('.../single/syn1/t4.mat')
data5 = sio.loadmat('.../single/syn1/t5.mat')
data6 = sio.loadmat('.../single/syn1/t6.mat')

abu0 = torch.from_numpy(data1['abu_est']).float()
abu1 = torch.from_numpy(data2['abu_est']).float()
abu2 = torch.from_numpy(data3['abu_est']).float()
abu3 = torch.from_numpy(data4['abu_est']).float()
abu4 = torch.from_numpy(data5['abu_est']).float()
abu5 = torch.from_numpy(data6['abu_est']).float()

abu = torch.stack([abu0,abu1, abu2, abu3, abu4, abu5])

def endmember(HSI):
    E_torch, _ = vca(HSI, P, snr_input=30)
    return E_torch

def End_deal(HSI):
    endmamber = np.empty((L, P,T))
    endmamber1 = torch.ones(L, P,T)

    for i in range(T):
        endmamber[:,:,i] = endmember(HSI[:, :, :,i].reshape(L, -1))
        endmamber1[:,:,i] = torch.from_numpy(endmamber[:,:,i])
        endmamber1[endmamber1<0] = 0

    return endmamber1

def min_max(x):
    # x: numpy array
    x = (x - torch.min(x)) / (torch.max(x) - torch.min(x))
    return x

def Nuclear_norm(inputs):
    band, h, w = inputs.shape
    input = torch.reshape(inputs, (band, h*w))
    out = torch.norm(input, p='nuc')
    return out

class SparseKLloss(nn.Module):
        def __init__(self):
            super(SparseKLloss, self).__init__()

        def __call__(self, input, decay=4e-4):
            input = torch.sum(input, 0, keepdim=True)
            loss = Nuclear_norm(input)
            return decay*loss

class NonZeroClipper(object):
    def __call__(self, module):
        if hasattr(module, 'weight'):
            w = module.weight.data
            w.clamp_(1e-6, 1)

pdf = loadmat('.../code_Cog-TD/syn1_copula.mat')
joint_pdf = torch.from_numpy(pdf['joint_PDF'])

#sample = loadmat('/home/home_node6_1/lry/MTHU/202503copula/result/copula_0422/tahoe_sample.mat')
#real_sample = torch.from_numpy(sample['sample'])

mat_contents1 = loadmat('.../dataset/synth_dataset_ex1.mat')
image = torch.from_numpy(mat_contents1['Y']) #[224,50,50,6]
HSI = image.contiguous().view(L, nr1, nc1, T).float()
#E = End_deal(HSI)
#print(E.shape,"E") torch.Size([173, 3, 6])
A_true = torch.from_numpy(mat_contents1['A']) 
de_ini = sio.loadmat('.../VCA-synth_ex1.mat')
E = torch.from_numpy(de_ini['Mn_hat_VRNN'][:,:,0,:]).squeeze()#(173, 3, 6)
ini_1 = E[:,:,0]
ini_2 = E[:,:,1]
ini_3 = E[:,:,2]
ini_4 = E[:,:,3]
ini_5 = E[:,:,4]
ini_6 = E[:,:,5]

E_T = torch.transpose(E, 0, 1)

l_vca = 0.01
l_2 = 0
use_bias = False
activation_set = nn.LeakyReLU(0.2)
initializer = torch.nn.init.xavier_normal_
re_loss = nn.MSELoss(reduction='mean')


# 自定义正则化函数
def E_reg(weight_matrix):
    return l_vca * torch.sum(torch.abs(weight_matrix - E_T)) + l_2 * torch.mean(torch.matmul(weight_matrix.T, weight_matrix))

import torch
import numpy as np

def sample_from_joint_distribution(joint_pdf, n_samples):
    """
    从联合分布中采样。
    :param joint_pdf: 联合概率密度函数，形状为 [6, 49500]
    :param n_samples: 每个维度的采样数量
    :return: 采样结果，形状为 [6, n_samples]
    """
    num_points = 7500
    x_points = torch.linspace(0, 1, num_points)
    delta_x = x_points[1] - x_points[0]  # 假设等间距

    # 计算每个维度的CDF
    cdf = torch.cumsum(joint_pdf * delta_x, dim=1)
    cdf /= cdf[:, -1].unsqueeze(1)  # 归一化

    # 生成均匀分布的随机数，形状为 [6, n_samples]
    uniform_samples = torch.rand(6, n_samples)

    # 逆变换采样（向量化实现）
    # 扩展cdf和uniform_samples以便广播
    cdf_expanded = cdf.unsqueeze(2)  # [6, 49500, 1]
    uniform_expanded = uniform_samples.unsqueeze(2)  # [6, n_samples,1]

    # 寻找插入位置
    indices = torch.searchsorted(cdf_expanded, uniform_expanded).squeeze(1)

    # 确保索引在有效范围内
    indices = torch.clamp(indices, 0, num_points - 1)

    # 收集对应的x值
    samples = x_points[indices]

    return samples


# 自定义层
class SparseReLU(nn.Module):
    def __init__(self):
        super(SparseReLU, self).__init__()
        self.zero = nn.Parameter(torch.zeros(1))
    
    def forward(self, x):
        return torch.where(x < 0, self.zero, x)
            
class SumToOne(nn.Module):
    def __init__(self):
        super(SumToOne, self).__init__()
    
    def forward(self, x):
        return x / torch.sum(x, dim=1, keepdim=True)

# 模型定义
class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim, 128,kernel_size=(1,1), stride=1, padding=(0,0)),
            nn.BatchNorm2d(128,momentum=0.9),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Conv2d(128, 64,kernel_size=(1,1), stride=1, padding=(0,0)),
            nn.BatchNorm2d(64,momentum=0.9),
            nn.ReLU(),
            nn.Conv2d(64, latent_dim, kernel_size=(1,1), stride=1, padding=(0,0)),
            nn.BatchNorm2d(latent_dim,momentum=0.9),
            nn.Softmax(dim=1)
        )
        self.sparse_relu = SparseReLU() 
        self.sum_to_one = SumToOne()  
    @staticmethod
    def weights_init(m):
        if type(m) == nn.Conv2d:
            nn.init.kaiming_normal_(m.weight.data)
    
    def forward(self, x):
        #x = x.view(T, L, nr1*nc1)
        x = self.encoder(x)
        results = []
        for i in range(x.shape[0]):
            # 提取每个三维子张量
            sub_x = x[i, :, :, :]  # 形状为 [3, 150, 110]
            # 应用 SparseReLU 和 SumToOne
            sub_x = self.sparse_relu(sub_x)
            #sub_x = self.sum_to_one(sub_x)
            results.append(sub_x)
        # 将结果组合成一个四维张量
        abu = torch.stack(results)
        abu = self.sum_to_one(abu)
        return abu

class Decoder(nn.Module):
    def __init__(self, latent_dim, input_dim):
        super(Decoder, self).__init__()
        self.decoder1 = nn.Sequential(
            nn.Linear(latent_dim, input_dim, bias=False)
            )
        self.decoder2 = nn.Sequential(
            nn.Linear(latent_dim, input_dim, bias=False)
            )
        self.decoder3 = nn.Sequential(
            nn.Linear(latent_dim, input_dim, bias=False)
            )
        self.decoder4 = nn.Sequential(
            nn.Linear(latent_dim, input_dim, bias=False)
            )
        self.decoder5 = nn.Sequential(
            nn.Linear(latent_dim, input_dim, bias=False)
            )
        self.decoder6 = nn.Sequential(
            nn.Linear(latent_dim, input_dim, bias=False)
            )
        
    @staticmethod
    def weights_init(m):
        if type(m) == nn.Conv2d:
            nn.init.kaiming_normal_(m.weight.data)
    
    def forward(self, x):
        
        out1 = self.decoder1(torch.transpose(x[0,:,:], 1, 0))
        out2 = self.decoder2(torch.transpose(x[1,:,:], 1, 0))
        out3 = self.decoder3(torch.transpose(x[2,:,:], 1, 0))
        out4 = self.decoder4(torch.transpose(x[3,:,:], 1, 0))
        out5 = self.decoder5(torch.transpose(x[4,:,:], 1, 0))
        out6 = self.decoder6(torch.transpose(x[5,:,:], 1, 0))
        re_out = torch.stack([out1, out2, out3, out4, out5, out6])
        re_out = re_out.permute(0, 2, 1).reshape(T, L, nr1, nc1)
        return re_out
        
class Discriminator(nn.Module):
    def __init__(self, latent_dim):
        super(Discriminator, self).__init__()
        self.layer1 = nn.Conv2d(latent_dim, 16,kernel_size=(1,1), stride=1, padding=(0,0))
        self.layer2 = nn.Conv2d(16, 32,kernel_size=(1,1), stride=1, padding=(0,0))
        self.layer3 = nn.Conv2d(32, 64,kernel_size=(1,1), stride=1, padding=(0,0))
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, 1)
    
    @staticmethod
    def weights_init(m):
        if type(m) == nn.Conv2d:
            nn.init.kaiming_normal_(m.weight.data)
    
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.sigmoid(self.layer3(x))
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)        
        # 全连接层
        x = self.sigmoid(self.fc(x))
        return x




# 损失函数
def SAD(output, target):
    return torch.sum(torch.abs(output - target)) / T*L*nc1*nr1

# 训练函数
def train(HSI):
    batch_size = T
    n_epochs = 200
    input_dim = L
    latent_dim = P
    encoder = Encoder(input_dim, latent_dim)
    decoder = Decoder(latent_dim, input_dim)
    discriminator = Discriminator(latent_dim)
    criterionSparse = SparseKLloss()
    apply_clamp_inst1 = NonZeroClipper()
    
    optimizer_ae = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()) , lr=1e-4)
    optimizer_d = optim.Adam(discriminator.parameters(), lr=1e-4)
    optimizer_g = optim.Adam(encoder.parameters(), lr=1e-4)

    encoder.apply(encoder.weights_init)

    model_dict = decoder.state_dict()
    model_dict['decoder1.0.weight'] = ini_1
    model_dict['decoder2.0.weight'] = ini_2
    model_dict['decoder3.0.weight'] = ini_3
    model_dict['decoder4.0.weight'] = ini_4
    model_dict['decoder5.0.weight'] = ini_5
    model_dict['decoder6.0.weight'] = ini_6
    decoder.load_state_dict(model_dict)

    discriminator.apply(discriminator.weights_init)

    time_start = time.time()
    for epoch in range(n_epochs):
        autoencoder_losses = []
        discriminator_losses = []
        generator_losses = []
        
        samples = HSI.permute(3,0,1,2).float()# T L nr nc
        #print(type(samples))
            
        # 训练自编码器
                    
        optimizer_ae.zero_grad()

        latent = encoder(samples)
        # 张量a
        #latent = latent / torch.sum(latent, dim=1, keepdim=True)
        #latent = (latent - 0.4) / 0.3
        #latent = torch.where(latent < 0, 0, latent)
        abundance = latent.reshape(T,P,nr1*nc1)
        reconstructed = decoder(abundance)

        E_pre1 = decoder.decoder1[0].weight.squeeze()
        E_loss1 = re_loss(E_pre1, E[:,:,0])
        E_pre2 = decoder.decoder2[0].weight.squeeze()
        E_loss2 = re_loss(E_pre2, E[:,:,1])
        E_pre3 = decoder.decoder3[0].weight.squeeze()
        E_loss3 = re_loss(E_pre3, E[:,:,2])
        E_pre4 = decoder.decoder4[0].weight.squeeze()
        E_loss4 = re_loss(E_pre4, E[:,:,3])
        E_pre5 = decoder.decoder5[0].weight.squeeze()
        E_loss5 = re_loss(E_pre5, E[:,:,4])
        E_pre6 = decoder.decoder6[0].weight.squeeze()
        E_loss6 = re_loss(E_pre6, E[:,:,5])
        E_loss = (E_loss1 + E_loss2 + E_loss3 + E_loss4 + E_loss5 + E_loss6)/6

        sparse = []
        for i in range(T):
            sparse.append(criterionSparse(latent[i,:,:,:].squeeze()))
        loss_sparse = sum(sparse)/T

        re_loss1 = re_loss(reconstructed, samples)
        sad_loss = SAD(reconstructed, samples)/(T*L*nc1*nr1)
        sad_loss = 200*sad_loss/(T*L*nc1*nr1)
        
        loss_ae = 20*re_loss1+sad_loss+1.15*E_loss+8*loss_sparse
        loss_ae.backward()
        optimizer_ae.step()

        decoder.decoder1.apply(apply_clamp_inst1)
        decoder.decoder2.apply(apply_clamp_inst1)
        decoder.decoder3.apply(apply_clamp_inst1)
        decoder.decoder4.apply(apply_clamp_inst1)
        decoder.decoder5.apply(apply_clamp_inst1)
        decoder.decoder6.apply(apply_clamp_inst1)

        autoencoder_losses.append(loss_ae.item())
            
        
        optimizer_d.zero_grad()
        fake_latent = latent.detach()
        #real_sample = torch.tensor(np.random.multivariate_normal(mean=mean, cov=cov, size=batch_size), dtype=torch.float32)

        real_sample = sample_from_joint_distribution(joint_pdf, P*nc1*nr1)
        real_sample = real_sample.reshape(T,P,nr1,nc1)
        #real_sample = abu.reshape(T,P,nr1,nc1)

        #real_sample_sort = real_sample
        #real_sample_sort[:,0,:] = real_sample[:,2,:]
        #real_sample_sort[:,1,:] = real_sample[:,2,:]
        #real_sample_sort[:,2,:] = real_sample[:,0,:]

        discriminator_input = torch.cat((fake_latent, real_sample))
        #print(discriminator_input.shape) 12, 3, 150, 110
        discriminator_labels = torch.cat((torch.zeros(batch_size, 1), torch.ones(batch_size, 1)))
        outputs = discriminator(discriminator_input)
        loss_d = nn.BCELoss()(outputs, discriminator_labels)
        loss_d.backward()
        optimizer_d.step()
        discriminator_losses.append(loss_d.item())
                
        # 训练生成器
        optimizer_g.zero_grad()
        latent = encoder(samples)
        outputs = discriminator(latent)
        loss_g = nn.BCELoss()(outputs, torch.ones(batch_size, 1))
        loss_g.backward()
        optimizer_g.step()
        generator_losses.append(loss_g.item())
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}")
            print(f"Autoencoder Loss: {np.mean(autoencoder_losses)}")
            print(f"Reconstruction Loss: {20*re_loss1}")
            print(f"sad Loss: {sad_loss}")
            print(f'E_loss:{1.15*E_loss}')
            print(f'loss_sparse:{5.05*loss_sparse}')
            print(f"Discriminator Loss: {np.mean(discriminator_losses)}")
            print(f"Generator Loss: {np.mean(generator_losses)}")        

            A_hat = latent.reshape(T,P,nr1,nc1)
            for t in range(T):
                abut = A_hat[t,:,:,:].squeeze()
                abu_est1 = abut/(torch.sum(abut, dim=0))
                #abu_est1 = min_max(abu_est1)
                #print('abut',abut.shape)#[3, 50, 50]
                abu_sample = real_sample[t,:,:].reshape(P,nr1,nc1)
                abu_true = A_true[:,:,:,t].squeeze()
                #print('abu_true',abu_true.shape)
                for i in range(P):
                    plt.subplot(3, P, i+1)
                    plt.imshow(abu_est1[i,:,:].detach().numpy())
                for i in range(P):
                    plt.subplot(3, P, P+i+1)
                    plt.imshow(abu_sample[i,:,:].detach().numpy())
                for i in range(P):
                    plt.subplot(3, P, 2*P+i+1)
                    plt.imshow(abu_true[i,:,:].detach().numpy())

                plt.show()
                plt.savefig('.../syn1'+'epoch'+str(epoch+1)+'time'+ str(t) +str(i)+"A.jpg")
        
        '''if (epoch + 1) % 200 == 0:
            torch.save(encoder.state_dict(), f'.../encoder_epoch_{epoch+1}.pth')
            torch.save(decoder.state_dict(), f'.../decoder_epoch_{epoch+1}.pth')
            torch.save(discriminator.state_dict(), f'.../discriminator_epoch_{epoch+1}.pth')'''
    time_end = time.time()
    total_time = time_end - time_start
    print('The training lasts for {} seconds'.format(total_time))
    
    model_dict = decoder.state_dict()
    
    
    t1 = model_dict['decoder1.0.weight'].reshape(L, P)
    t2 = model_dict['decoder2.0.weight'].reshape(L, P)
    t3 = model_dict['decoder3.0.weight'].reshape(L, P)
    t4 = model_dict['decoder4.0.weight'].reshape(L, P)
    t5 = model_dict['decoder5.0.weight'].reshape(L, P)
    t6 = model_dict['decoder6.0.weight'].reshape(L, P)

    N = nr1*nc1
    Mn_hat = torch.zeros((L, P, N, T))
    
    for i in range(N):
        Mn_hat[:, :, i, 0] = t1.clamp_(0,1)
    for i in range(N):
        Mn_hat[:, :, i, 1] = t2.clamp_(0,1)
    for i in range(N):
        Mn_hat[:, :, i, 2] = t3.clamp_(0,1)
    for i in range(N):
        Mn_hat[:, :, i, 3] = t4.clamp_(0,1)
    for i in range(N):
        Mn_hat[:, :, i, 4] = t5.clamp_(0,1)
    for i in range(N):
        Mn_hat[:, :, i, 5] = t6.clamp_(0,1)
        
    E_pre = Mn_hat

    return latent, E_pre


# 主函数
if __name__ == "__main__":
    
    latent, endmembers = train(HSI)
    #endmembers = endmembers.reshape(T,L,P)
    A_hat = latent.reshape(T,P,nr1,nc1)

    
    

    matpath = '.../syn1.mat'
    sio.savemat(matpath, \
                    {'A_hat' : A_hat.cpu().detach().numpy(),
                    'Mn_hat' : endmembers.cpu().detach().numpy(),
                    'M_VCA': E.detach().numpy()})
