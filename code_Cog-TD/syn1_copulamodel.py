import torch
import torch.nn as nn
from torch import optim
import numpy as np
from scipy.stats import gaussian_kde
import pandas as pd
from scipy.stats import rankdata
import torch.nn.functional as F
import torch.autograd as autograd
import math
from sklearn.decomposition import PCA
import numpy as np
from sklearn.random_projection import GaussianRandomProjection


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class marginal(nn.Module):
    def __init__(self, input_dim, hidden_width, num_hidden):
        super().__init__()
        self.layers = nn.ModuleList()
        in_dim = input_dim
        for _ in range(num_hidden):
            self.layers.append(nn.Linear(in_dim, hidden_width))
            in_dim = hidden_width
        self.final = nn.Linear(in_dim, 1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x.requires_grad_(True)  # Enable gradient tracking
        identity = x  # Save original input for gradient calculation
        
        # Forward pass through hidden layers
        for layer in self.layers:
            x = torch.tanh(layer(x))
        
        # Final layer with sigmoid activation
        cdf = torch.sigmoid(self.final(x))
        
        # Calculate PDF using automatic differentiation
        pdf = autograd.grad(
            outputs=cdf,
            inputs=identity,
            grad_outputs=torch.ones_like(cdf),
            create_graph=True,
            retain_graph=True
        )[0]
        
        # Apply ReLU and numerical stability
        neg = self.relu(-pdf)
        pdf = 1e-9 + self.relu(pdf)
        
        return cdf, pdf, neg

class copula(nn.Module):
    def __init__(self, input_dim, hidden_width, num_hidden):
        super().__init__()
        self.layers = nn.ModuleList()
        in_dim = input_dim
        for _ in range(num_hidden):
            self.layers.append(nn.Linear(in_dim, hidden_width))
            in_dim = hidden_width
        self.final = nn.Linear(in_dim, 1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x.requires_grad_(True)  # Enable gradient tracking
        components = [x[:, i:i+1] for i in range(x.size(1))]
        
        # Forward pass through hidden layers
        h = torch.cat(components, dim=1)
        for layer in self.layers:
            h = torch.tanh(layer(h))
        
        # Final layer with sigmoid activation
        cdf = torch.sigmoid(self.final(h))
        
        # Calculate mixed partial derivatives
        pdf = cdf
        for comp in components:
            pdf = autograd.grad(
                outputs=pdf,
                inputs=comp,
                grad_outputs=torch.ones_like(pdf),
                create_graph=True,
                retain_graph=True
            )[0]
        
        # Apply ReLU and numerical stability
        neg = self.relu(-pdf)
        pdf = 1e-9 + self.relu(pdf)
        
        return cdf, pdf, neg
            
class JointModel(nn.Module):
    """修改后的joint模型"""
    def __init__(self):
        super().__init__()
        self.num_endmembers = 3
        self.spectral_dim = 224
        self.height = 50
        self.weight = 50
        self.num_temporal = 6

        # 输入数据的维度。作用：定义了模型的输入特征数量
        self.number_of_dimension = self.num_endmembers*self.height*self.weight
        #self.input_dim = self.num_endmembers*self.height*self.weight
        # 边际分布模型中每个隐藏层的神经元数量
        self.hidden_layer_width_for_marginal = 32
        # 边际分布模型中的隐藏层数量
        self.num_hidden_for_marginal = 5
        self.hidden_layer_width_for_copula = 32
        self.num_hidden_for_copula = 5

        self.marginal_models = nn.ModuleList ([
            marginal(
                input_dim= 1,
                hidden_width=self.hidden_layer_width_for_marginal,
                num_hidden=self.num_hidden_for_marginal
                ) for _ in range(self.num_temporal)
                ]).to("cuda")
        
        self.copula_model = copula(
            input_dim=self.num_temporal,
            hidden_width=self.hidden_layer_width_for_copula,
            num_hidden= self.num_hidden_for_copula
            ).to("cuda")
        
    @staticmethod
    def weights_init(m):
        if type(m) == nn.Conv2d:
            nn.init.kaiming_normal_(m.weight.data)

    def forward(self, x):
        abu = x.to("cuda")#torch.Size([6, 49500])
        abu = abu.reshape(self.num_temporal, self.number_of_dimension)
        
        marginal_inputs = [abu[_,:] for _ in range(self.num_temporal)]
        #print('marginal_inputs',len(marginal_inputs),len(marginal_inputs[0])) 
        #torch.Size([49500, 1])6 49500

        # Get predictions from marginal models
        marginal_cdfs = []
        marginal_pdfs = []
        marginal_negs = []

        for i in range(len(marginal_inputs)):
            
            transposed_list = [[marginal_inputs[i][j]] for j in range(len(marginal_inputs[0]))]
            inputs = torch.tensor(transposed_list).to("cuda")   
            #print('inputs',inputs.shape) #[49500, 1]
            cdf, pdf, neg = self.marginal_models[i](inputs)
            # CDF本质是对张量求和
            marginal_cdfs.append(cdf)
            marginal_pdfs.append(pdf)
            marginal_negs.append(neg)

        #copula_input = abu.transpose(0, 1)
        m_CDF_c = torch.stack(marginal_cdfs).squeeze().to("cuda") #6 2500
        #print('m_CDF_c',m_CDF_c.shape) m_CDF_c torch.Size([6, 2500])
        m_PDF_c = torch.stack(marginal_pdfs).squeeze().to("cuda")
        #print('m_PDF_c',m_PDF_c.shape) 
        #a_m_PDF_c = torch.mean(m_PDF_c, dim=2, keepdim=True)
        #print('a_m_PDF_c',m_PDF_c.shape) a_m_PDF_c torch.Size([6, 16500, 3])

        pca = PCA(n_components=6)  # 降维到6维
        m_CDF_reduced = pca.fit_transform(m_CDF_c.cpu().detach().numpy())
        m_CDF_reduced = torch.tensor(m_CDF_reduced).to("cuda")
        #print('m_CDF_reduced',m_CDF_reduced.shape) #m_CDF_reduced torch.Size([6, 6])
        #m_CDF = m_CDF_reduced.transpose(0, 1)

        #copula_input = torch.stack(marginal_cdfs).transpose(0, 1)
        # print('copula_input',copula_input.shape) copula_input torch.Size([49500, 6])
        copula_cdf, copula_pdf, copula_neg = self.copula_model(m_CDF_reduced)
        #print('copula_outputs1',copula_cdf.shape) 
        #copula_outputs1 torch.Size([6, 1])
        marginals = m_PDF_c
        #print('marginals',marginals.shape) 
        p_joint = copula_pdf*marginals
        # print('p_joint',p_joint.shape) p_joint torch.Size([6, 16500])

        c_CDF_c = copula_cdf.squeeze().to("cuda")
        c_PDF_c = copula_pdf.squeeze().to("cuda")

        j_PDF_c = p_joint

        
        return m_CDF_c, m_PDF_c, c_CDF_c, c_PDF_c, j_PDF_c