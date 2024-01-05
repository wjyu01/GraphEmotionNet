import torch
import torch.nn as nn
import torch.nn.functional as F
import adv_layer
import math
import random

class TemporalAttention(nn.Module):
    '''
    compute temporal attention scores
    --------
    Input:  (batch_size, num_of_timesteps, num_of_vertices, num_of_features)
    Output: (batch_size, num_of_timesteps, num_of_timesteps)
    '''
    def __init__(self, device, num_of_timesteps, num_of_vertices, num_of_features):
        super(TemporalAttention, self).__init__()
        self.U1 = nn.init.normal_(nn.Parameter(torch.FloatTensor(num_of_vertices).to(device)))       
        self.U2 = nn.init.normal_(nn.Parameter(torch.FloatTensor(num_of_features, num_of_vertices).to(device)))
        self.U3 = nn.init.normal_(nn.Parameter(torch.FloatTensor(num_of_features).to(device)))
        self.be = nn.init.normal_(nn.Parameter(torch.FloatTensor(1, num_of_timesteps, num_of_timesteps).to(device)))
        self.Ve = nn.init.normal_(nn.Parameter(torch.FloatTensor(num_of_timesteps, num_of_timesteps).to(device)))
    def forward(self, x):
        '''
        :param x: (batch_size, T, V, F)
        :return: (B, T, T)
        '''
        lhs = torch.matmul(torch.matmul(x.permute(0,1,3,2), self.U1), self.U2)#(bs,3,62: T,V)
        rhs = torch.matmul(self.U3, x.permute(0,2,3,1))  # (F)(B,V,F,T)->(B, V, T)
        product = torch.matmul(lhs, rhs)  # (B,T,V)(B,V,T)->(B,T,T)
        E = torch.matmul(self.Ve, torch.sigmoid(product + self.be))  # (B, T, T)
        E_normalized = F.softmax(E, dim=1)
        return E_normalized
  
def diff_loss(diff, S, Falpha):
    '''
    compute the 1st loss of L_{graph_learning}
    '''
    if len(S.shape)==4:
        # batch input
        return Falpha * torch.mean(torch.sum(torch.sum(diff**2,axis=3)*S, axis=(1,2)))
    else:
        return Falpha * torch.sum(torch.matmul(S,torch.sum(diff**2,axis=2)))
  
def F_norm_loss(S, Falpha):
    '''
    compute the 2nd loss of L_{graph_learning}
    '''
    if len(S.shape)==3:
        # batch input
        return Falpha * torch.sum(torch.mean(S**2,axis=0))
    else:
        return Falpha * torch.sum(S**2)

class Graph_Learn(nn.Module):
    '''
    Graph structure learning (based on the middle time slice)
    --------
    Input:  (batch_size, num_of_timesteps, num_of_vertices, num_of_features)
    Output: (batch_size, num_of_vertices, num_of_vertices)
    '''
    def __init__(self,alpha, num_of_features, device):
        super(Graph_Learn, self).__init__()
        self.alpha = alpha
        self.a = nn.init.uniform_(nn.Parameter(torch.FloatTensor(num_of_features, 1).to(device)))
        self.S = torch.zeros(1,1,1,1)  # similar to placeholder
        self.diff = torch.zeros(1,1,1,1,1)  # similar to placeholder

    def forward(self, x):
        N, T, V, f = x.shape
        # shape: (N,V,F) use the current slice (middle one slice)
        x = x[:,int(x.shape[1])//2,:,:]
        # shape: (N,V,V,F)
        diff = (x.expand(V,N,V,f).permute(2,1,0,3)-x.expand(V,N,V,f)).permute(1,0,2,3)#62*61+62
        # shape: (N,V,V)
        tmpS = torch.exp(F.relu(torch.reshape(torch.matmul(torch.abs(diff), self.a), [N,V,V])))
        # normalization
        S = tmpS / torch.sum(tmpS,axis=1,keepdims=True)
        self.diff = diff
        self.S = S
        Sloss = F_norm_loss(self.S,self.alpha)
        dloss = diff_loss(self.diff,self.S,self.alpha)
        return S,Sloss,dloss

class SpatialAttention(nn.Module):
    '''
    compute spatial attention scores
    --------
    Input:  (batch_size, num_of_timesteps, num_of_vertices, num_of_features)
    Output: (batch_size, num_of_vertices, num_of_vertices)
    '''
    def __init__(self, device, num_of_timesteps, num_of_vertices, num_of_features):
        super(SpatialAttention, self).__init__()
        self.W1 = nn.init.normal_(nn.Parameter(torch.FloatTensor(num_of_timesteps).to(device)))
        self.W2 = nn.init.normal_(nn.Parameter(torch.FloatTensor(num_of_features, num_of_timesteps).to(device)))
        self.W3 = nn.init.normal_(nn.Parameter(torch.FloatTensor(num_of_features).to(device)))
        self.bs = nn.init.normal_(nn.Parameter(torch.FloatTensor(1, num_of_vertices,num_of_vertices).to(device)))
        self.Vs = nn.init.normal_(nn.Parameter(torch.FloatTensor(num_of_vertices, num_of_vertices).to(device)))
    def forward(self, x):
        '''
        :param x: (batch_size, T, V, F)
        :return: (B,N,N)
        '''
        lhs = torch.matmul(torch.matmul(x.permute(0,2,3,1), self.W1), self.W2) #(bs,62,3: V,T)
        rhs = torch.matmul(self.W3, x.permute(0,1,3,2))  #(bs,3,62: T,V)
        product = torch.matmul(lhs, rhs)  # (b,V,T)(b,T,V) -> (B, V, V)
        S = torch.matmul(self.Vs, torch.sigmoid(product + self.bs))  # (V,V)(B, V, V)->(B,V,V)
        S_normalized = F.softmax(S, dim=1)
        return S_normalized

class cheb_conv_with_SAt_GL(nn.Module):
    '''
    K-order chebyshev graph convolution after Graph Learn
    --------
    Input:  [x   (batch_size, num_of_timesteps, num_of_vertices, num_of_features),
             SAtt(batch_size, num_of_vertices, num_of_vertices),
             S   (batch_size, num_of_vertices, num_of_vertices)]
    Output: (batch_size, num_of_timesteps, num_of_vertices, num_of_filters)
    '''
    def __init__(self, num_of_filters, k, num_of_features, device):
        super(cheb_conv_with_SAt_GL, self).__init__()
        self.Theta = nn.ParameterList([nn.init.uniform_(nn.Parameter(torch.FloatTensor(num_of_features, num_of_filters).to(device))) for _ in range(k)])
        self.out_channels = num_of_filters
        self.K = k
        self.device = device
        
    def forward(self, x):
        #Input:  [x,SAtt,S]
        assert isinstance(x, list)
        assert len(x)==3,'cheb_conv_with_SAt_GL: number of input error'
        x, spatial_attention, W = x
        N, T, V, f = x.shape
        #Calculating Chebyshev polynomials
        D = torch.diag_embed(torch.sum(W,axis=1))
        L = D - W
        '''
        Here we approximate λ_{max} to 2 to simplify the calculation.
        For more general calculations, please refer to here:
            lambda_max = K.max(tf.self_adjoint_eigvals(L),axis=1)
            L_t = (2 * L) / tf.reshape(lambda_max,[-1,1,1]) - [tf.eye(int(num_of_vertices))]
        '''
        
        lambda_max = 2.0
        L_t =( (2 * L) / lambda_max - torch.eye(int(V)).to(self.device))
        cheb_polynomials = [torch.eye(int(V)).to(self.device), L_t]
        for i in range(2, self.K):
            cheb_polynomials.append(2 * L_t * cheb_polynomials[i - 1] - cheb_polynomials[i - 2])
        #Graph Convolution
        outputs = []
        for time_step in range(T):
            graph_signal = x[:, time_step, :, :]  # (b, V, F_in)
            output = torch.zeros(N, V, self.out_channels).to(self.device)  # (b, V, F_out)
            for k in range(self.K):
                T_k = cheb_polynomials[k]  # (V,V)
                T_k_with_at = T_k.mul(spatial_attention) # (V,V)*(V,V) = (V,V) 多行和为1, 按着列进行归一化
                theta_k = self.Theta[k]  # (in_channel, out_channel)
                rhs = T_k_with_at.permute(0, 2, 1).matmul(graph_signal)   # (V, V)(b, V, F_in) = (b, V, F_in) 因为是左乘，所以多行和为1变为多列和为1，即一行之和为1，进行左乘
                output = output + rhs.matmul(theta_k)  # (b, V, F_in)(F_in, F_out) = (b, V, F_out)
            outputs.append(output.unsqueeze(1))  # (b, 1, V, F_out)
        return F.relu(torch.cat(outputs, dim=1))  # (b, T, V, F_out)

class cheb_conv_withSAt(nn.Module):
    '''
    K-order chebyshev graph convolution
    '''
    def __init__(self, K, cheb_polynomials, in_channels, out_channels):
        '''
        :param K: int
        :param in_channles: int, num of channels in the input sequence
        :param out_channels: int, num of channels in the output sequence
        '''
        super(cheb_conv_withSAt, self).__init__()
        self.K = K
        self.cheb_polynomials = cheb_polynomials
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.DEVICE = torch.device('cuda:0')#torch.device('cuda:0')#cheb_polynomials[0].device
        self.Theta = nn.ParameterList([nn.Parameter(torch.FloatTensor(in_channels, out_channels).to(self.DEVICE)) for _ in range(K)])
    
    def forward(self, x, spatial_attention):
        '''
        Chebyshev graph convolution operation
        :param x: (batch_size, T, V, F_in)
        :return: (batch_size, T, ,V,F_out)
        '''
        batch_size, num_of_timesteps, num_of_vertices, num_of_features = x.shape
        outputs = []
        for time_step in range(num_of_timesteps):
            graph_signal = x[:, time_step, :, :]  # (b, V, F_in)
            output = torch.zeros(batch_size, num_of_vertices, self.out_channels).to(self.DEVICE)  # (b, V, F_out)
            for k in range(self.K):
                T_k = self.cheb_polynomials[k]  # (V,V)
                T_k_with_at = T_k.mul(spatial_attention) # (V,V)*(V,V) = (V,V) 多行和为1, 按着列进行归一化
                theta_k = self.Theta[k]  # (in_channel, out_channel)
                rhs = T_k_with_at.permute(0, 2, 1).matmul(graph_signal)  # (V, V)(b, V, F_in) = (b, V, F_in) 因为是左乘，所以多行和为1变为多列和为1，即一行之和为1，进行左乘
                output = output + rhs.matmul(theta_k)  # (b, V, F_in)(F_in, F_out) = (b, V, F_out)
            outputs.append(output.unsqueeze(1))  # (b, 1, V, F_out)
        return F.relu(torch.cat(outputs, dim=1))  # (b, T, V, F_out)

class STGCN_block(nn.Module):

    def __init__(self, net_params):
        super(STGCN_block, self).__init__()
        self.num_of_timesteps = net_params['num_of_timesteps']
        self.num_of_vertices = net_params['num_of_vertices']
        self.num_of_features = net_params['num_of_features']
        device = net_params['DEVICE']
        node_feature_hidden1 = net_params['node_feature_hidden1']
        node_feature_hidden2 = net_params['node_feature_hidden2']
        self.TAt = TemporalAttention(device, self.num_of_timesteps, self.num_of_vertices, self.num_of_features)
        self.SAt = SpatialAttention(device, self.num_of_timesteps, self.num_of_vertices, self.num_of_features)
        self.Graph_Learn = Graph_Learn(net_params['GLalpha'], self.num_of_features, device)
        self.cheb_conv_SAt_GL = cheb_conv_with_SAt_GL(node_feature_hidden1,net_params['K'], self.num_of_features, device)
        self.time_conv = nn.Conv2d(node_feature_hidden1, node_feature_hidden2, kernel_size=(1, 3), stride=(1, 3))
        self.residual_conv = nn.Conv2d(self.num_of_features, node_feature_hidden2, kernel_size=(1, 1), stride=(1, 3))
        self.ln = nn.LayerNorm(node_feature_hidden2)  #需要将channel放到最后一个维度上
        
    def forward(self, x):
        '''
         x: input(bs,T,V,F)
         return:(bs,T,V,F2)
        '''
        # TAt
        temporal_At = self.TAt(x)  # (b, T, T)
        x_TAt = torch.matmul((x.permute(0,2,3,1)).reshape(x.shape[0],-1,self.num_of_timesteps), temporal_At).reshape(x.shape[0], self.num_of_vertices, self.num_of_features, self.num_of_timesteps)
        x_TAt = x_TAt.permute(0,3,1,2)/math.sqrt(62*5)
        # SAt
        spatial_At = self.SAt(x_TAt)
        S,Sloss,dloss = self.Graph_Learn(x)
        spatial_gcn = self.cheb_conv_SAt_GL([x, spatial_At, S])
        # convolution along the time axis
        time_conv_output = self.time_conv(spatial_gcn.permute(0, 3, 2, 1))/math.sqrt(3*5)  # (b,T,V,F)->(b,F,V,T) 用(1,3)的卷积核去做->(b,F,V,T)
        # residual shortcut
        x_residual = self.residual_conv(x.permute(0, 3, 2, 1))  # (b,T,V,F)->(b,F,V,T) 用(1,3)的卷积核去做->(b,F,V,T)
        x_residual = self.ln(F.relu(x_residual + time_conv_output).permute(0, 3, 2, 1))# (b,F,V,T)->(b,T,V,F)
        x_residual = x_residual.squeeze(1)
        return x_residual,Sloss,dloss,S

class feature_extractor(nn.Module):
    def __init__(self,input, hidden_1,hidden_2):
         super(feature_extractor,self).__init__()
         self.fc1=nn.Linear(input,hidden_1)
         self.fc2=nn.Linear(hidden_1,hidden_2)
         self.dropout1 = nn.Dropout(p=0.25)
         self.dropout2 = nn.Dropout(p=0.25)
    def forward(self,x):
         x=self.fc1(x)
         x1=F.relu(x)
#         x=F.leaky_relu(x)
         x2=self.fc2(x1)
         x2=F.relu(x2)
#         x=F.leaky_relu(x)
         return x2



class STGCN_contrastNet(nn.Module):
    def __init__(self, net_params):
        super(STGCN_contrastNet, self).__init__()
        self.device = net_params['DEVICE']
        self.STGCN = STGCN_block(net_params)
        self.adv_alpha = net_params['adv_alpha']
        self.domain_classifier = adv_layer.DomainAdversarialLoss(hidden_1=64)
        self.aug_type = net_params['aug_type']
        #self.train_test = net_params['train']
        #self.head = net_params['projection_head']
        self.fea_extrator_f = feature_extractor(310, 64, 64)
        self.classifier_noproto = nn.Linear(64, 3)
        self.g_list = []
    def forward(self, x):

        ##时空图特征提取
        feature, Sloss, dloss, S = self.STGCN(x) #时空图特征提取
        feature1 = torch.flatten(feature, start_dim=1, end_dim=-1)
        feature1 = self.fea_extrator_f(feature1)
        pred = self.classifier_noproto(feature1)
        domain_output = self.domain_classifier(feature1)
        return pred, domain_output, Sloss, dloss
      
 














