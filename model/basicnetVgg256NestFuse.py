from tkinter.tix import Tree
import numpy as np
import math
import torch
from torch import nn
import torch.nn.functional as F
import model.resnet1 as res
from torch_geometric.nn import BatchNorm, global_max_pool, global_mean_pool
from model.graphbatch import graph_batch_trans_block4to1, graph_batch_trans_block4to2, graph_batch_trans_block_singlepatch, graph_batch_re_trans
from model.myGatedEdgeConv import Gatedgcn, GatedEdgeConv, AdGatedEdgeConv
from matplotlib import pyplot as plt
import math
import scipy.stats
from torch.utils.data import Dataset
import cv2
import os
# from timm.models.layers import DropPath, to_2tuple, trunc_normal_
# import model.vgg as vgg
import model.vgg512 as vgg

# 0~1 normalization.
def MaxMinNormalization(x):
    Max = torch.max(x)
    Min = torch.min(x)
    x = torch.div(torch.sub(x, Min), 0.0001 + torch.sub(Max, Min))
    return x


def weight_init(module):
    for n, m in module.named_children():
        print('initialize: '+n)
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, nn.ReLU):
            pass
        else:
            m.initialize()


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False, groups=groups)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class TransBasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=2, stride=2, padding=0, dilation=1, bias=False):
        super(TransBasicConv2d, self).__init__()
        self.Deconv = nn.ConvTranspose2d(in_planes, out_planes,
                                         kernel_size=kernel_size, stride=stride,
                                         padding=padding, dilation=dilation, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.Deconv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class NodeAtt(nn.Module):
    def __init__(self, in_channels):
        super(NodeAtt, self).__init__()
        self.mlp = nn.Sequential(nn.Linear(1 * in_channels, in_channels),
                                 nn.BatchNorm1d(in_channels))
        self.lin = nn.Sequential(nn.Linear(1 * in_channels, in_channels),
                                 nn.BatchNorm1d(in_channels))

    def forward(self, x):  # x has shape [N, 1*in_channels]
        max_out, _ = torch.max(self.mlp(x), dim=1, keepdim=True)  # NC→N1
        nodeatt = torch.sigmoid(max_out)  # has shape [N,1]
        x_out = self.lin(x * nodeatt) #+ x   # [N, in_channels]
        return x_out


class GraphConvNet(nn.Module):
    '''Simple GCN layer, similar to https://arxiv.org/abs/1609.02907'''
    def __init__(self, in_features, out_features, node_num, bias):
        super(GraphConvNet, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(1, out_features, node_num))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj):
        x_t = x.permute(0, 2, 1).contiguous()  # b x k x c
        support = torch.matmul(x_t, self.weight)  # b x k x c
        adj = torch.softmax(adj, dim=2)
        output = (torch.matmul(adj, support)).permute(0, 2, 1).contiguous()  # b x c x k
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'



class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return


class ParallGCNetPyg(nn.Module):
    def __init__(self, dim, loop, bknum, thr, dims):
        super(ParallGCNetPyg, self).__init__()
        self.bknum = bknum
        self.loop = loop
        self.gcn1 = gcn_metablock(dim)
        self.gcn2 = gcn_metablock(dim)

        self.relu = nn.ReLU()
        self.lineA1 = torch.nn.Sequential(torch.nn.Linear(dim, 1),
                                          torch.nn.BatchNorm1d(1),)
                                          # torch.nn.ReLU())
        self.lineA2 = torch.nn.Sequential(torch.nn.Linear(dim, 1),
                                          torch.nn.BatchNorm1d(1),)
                                          # torch.nn.ReLU())
        self.mlpCA1 = torch.nn.Sequential(torch.nn.Linear(dim, dim),
                                          torch.nn.BatchNorm1d(dim),
                                          torch.nn.ReLU(),
                                          torch.nn.Linear(dim, dim),
                                          torch.nn.BatchNorm1d(dim))
        self.mlpCA2 = torch.nn.Sequential(torch.nn.Linear(dim, dim),
                                          torch.nn.BatchNorm1d(dim),
                                          torch.nn.ReLU(),
                                          torch.nn.Linear(dim, dim),
                                          torch.nn.BatchNorm1d(dim))
        self.lineFu = torch.nn.Linear(dim, dim)

    def forward(self, x, adj):  # x is [B, C, node_num]
        B, C, node_num = x.size()
        bndnum = int(node_num / (self.bknum * self.bknum))

        x_t = x.permute(0, 2, 1).contiguous()  # x_t is [B, node_num, C]

        graph_data2, edge_index2, edge_attr2, B, graph_batch2 = graph_batch_trans_block_singlepatch(adj, bndnum, 0, B, x_t, node_num, self.bknum)  # x is [B, C, node_num], output is [node_num_batch, C], [2, node_connect_batch], [node_connect_batch]
        y2 = self.gcn2(graph_data2, edge_index2, edge_attr2)
        graph_data1, edge_index1, edge_attr1, B, graph_batch1 = graph_batch_trans_block4to1(adj, bndnum, 0, B, x_t, node_num, self.bknum)  # x is [B, C, node_num], output is [node_num_batch, C], [2, node_connect_batch], [node_connect_batch]
        y1 = self.gcn1(graph_data1, edge_index1, edge_attr1)

        w1 = torch.sigmoid(self.mlpCA1(global_max_pool(y1, graph_batch1)))
        w2 = torch.sigmoid(self.mlpCA2(global_max_pool(y2, graph_batch2)))  # B*C_new，channel-wise Attention
        a1 = torch.sigmoid(self.lineA1(global_mean_pool(y1, graph_batch1)))
        a2 = torch.sigmoid(self.lineA2(global_mean_pool(y2, graph_batch2)))  # B*1
        A1 = (a1 + (1-a2))/2.0
        A2 = (a2 + (1-a1))/2.0
        A = torch.cat((A2, A1), dim=1)
        A = F.softmax(A, dim=1)
        W2 = w2[graph_batch2]  # (B*node_num)*C_new
        A2 = A[:, 0][graph_batch2]  # (B*node_num)*1
        W1 = w1[graph_batch1]
        A1 = A[:, 1][graph_batch1]  # (B*node_num)*1
        y = self.lineFu(A2.unsqueeze(1) * W2 * y2 + A1.unsqueeze(1) * W1 * y1)
        node_num_batch, C_new = y.size()
        output = graph_batch_re_trans(y, B, node_num_batch, C_new)
        return output

class gcn_metablock(nn.Module):
    def __init__(self, dim):
        super(gcn_metablock, self).__init__()
        self.gcn = AdGatedEdgeConv(dim, dim)
        self.bn = BatchNorm(dim)
        self.Att = NodeAtt(dim)
        self.relu = nn.GELU()
        self.lin1 = torch.nn.Sequential(torch.nn.Linear(dim, 1*dim),
                                        torch.nn.BatchNorm1d(1*dim),
                                       torch.nn.GELU())
        self.lin2 = torch.nn.Sequential(torch.nn.Linear(dim, 1*dim),
                                        torch.nn.BatchNorm1d(1*dim),
                                        )

    def forward(self, graph_data, edge_index, edge_attr):
        x = self.lin1(graph_data)
        y = self.Att(self.relu(self.bn(self.gcn(x, edge_index, edge_attr))))
        y = self.lin2(y) + graph_data
        return y

class PositionBias(nn.Module):
    def __init__(self, h, w, k):
        super(PositionBias, self).__init__()
        self.h = h
        self.w = w
        self.k = k

        coords_h = torch.arange(self.h)
        coords_w = torch.arange(self.w)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.h - 1
        relative_coords[:, :, 1] += self.w - 1
        relative_coords[:, :, 0] *= 2 * self.w - 1
        relative_position_index = relative_coords.sum(-1)

        relative_position_index_larger = torch.zeros([self.h * self.h * self.k, self.w * self.w * self.k], dtype=int)
        for i in range(self.h * self.h):
            for j in range(self.w * self.w):
                relative_position_index_larger[i * self.k:(i + 1) * self.k, j * self.k:(j + 1) * self.k] = relative_position_index[i, j]


        self.relative_position_index = relative_position_index_larger


        self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * self.h - 1) * (2 * self.w - 1)))
        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self):
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(self.h * self.w * self.k, self.h * self.w * self.k, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        return relative_position_bias


class GraphNet(nn.Module):
    def __init__(self,bnum,bnod,dim,normalize_input=False):
        super(GraphNet, self).__init__()
        self.bnum=bnum
        self.bnod=bnod
        self.node_num=bnum*bnum*bnod
        self.dim = dim
        self.normalize_input = normalize_input
        self.anchor = nn.Parameter(torch.rand(self.node_num, dim))
        self.sigma = nn.Parameter(torch.rand(self.node_num, dim))
        self.posbias = PositionBias(self.bnum, self.bnum, self.bnod)


    def gen_soft_assign(self, x, sigma):
        B, C, H, W = x.size()
        soft_assign = torch.zeros([B, self.node_num, self.n], device=x.device, dtype=x.dtype, layout=x.layout)
        soft_ass = torch.zeros([B, self.node_num, self.n], device=x.device, dtype=x.dtype, layout=x.layout)
        for node_id in range(self.node_num):
            block_id=math.floor(node_id/self.bnod)
            h_sta=math.floor(block_id/self.bnum)*self.h
            w_sta=block_id%(self.bnum)*self.w
            h_end=h_sta+self.h
            w_end=w_sta+self.w
            tmp=x.view(B, C, H, W)[: , : ,h_sta:h_end , w_sta : w_end]
            tmp=tmp.reshape(B,C,-1).permute(0,2,1).contiguous()
            residual = (tmp - self.anchor[node_id, :]).div(sigma[node_id, :]) # + eps)
            soft_assign[:, node_id, :] = -torch.pow(torch.norm(residual, dim=2),2)/2

        for block_id in range(self.bnum*self.bnum):
            node_sta=self.bnod*block_id
            node_end=node_sta+self.bnod
            soft_ass[:, node_sta:node_end, :] = F.softmax(soft_assign[:, node_sta:node_end, :], dim=1)

        return soft_ass

    def forward(self, x):
        B, C, H, W = x.size()
        self.h=math.floor(H/self.bnum)
        self.w=math.floor(W/self.bnum)
        self.n=self.h*self.w
        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)
        sigma = torch.sigmoid(self.sigma)
        soft_assign = self.gen_soft_assign(x, sigma)
        eps = 1e-9
        nodes = torch.zeros([B, self.node_num, C], dtype=x.dtype, layout=x.layout, device=x.device)
        for node_id in range(self.node_num):
            block_id=math.floor(node_id/self.bnod)
            h_sta=math.floor(block_id/self.bnum)*self.h
            w_sta=block_id%(self.bnum)*self.w
            h_end=h_sta+self.h
            w_end=w_sta+self.w
            tmp=x.view(B, C, H, W)[: , : ,h_sta:h_end , w_sta : w_end]
            tmp=tmp.reshape(B,C,-1).permute(0,2,1).contiguous()
            residual = (tmp - self.anchor[node_id, :]).div(sigma[node_id, :])  # + eps)
            nodes[:, node_id, :] = residual.mul(soft_assign[:, node_id, :].unsqueeze(2)).sum(dim=1) / (soft_assign[:, node_id, :].sum(dim=1).unsqueeze(1) + eps)
        nodes = F.normalize(nodes, p=2, dim=2)
        nodes = nodes.view(B, -1).contiguous()  # B X (Node_num X C)
        nodes = F.normalize(nodes, p=2, dim=1)  # l2 normalize
        graph = nodes.view(B, self.node_num, C).permute(0, 2, 1).contiguous()  # BCN
        graph_t = graph.permute(0, 2, 1).contiguous()  # x_t is [B, node_num, C]
        posbias = self.posbias()
        adj = torch.matmul(graph_t, graph) + posbias  # adj is [B, node_num, node_num]
        adj = MaxMinNormalization(adj)
        return graph, adj, soft_assign

    def initKmensBlock(self, initdata): #initdata is B C H W
        # 读取原始图像
        C, H, W = initdata[0].size()
        self.h = math.floor(H / self.bnum)
        self.w = math.floor(W / self.bnum)
        self.n = self.h * self.w
        anchor = np.zeros((self.node_num, C))
        sigma = np.zeros((self.node_num, C))
        for block_id in range(self.bnum * self.bnum):
            h_sta = math.floor(block_id / self.bnum) * self.h
            w_sta = block_id % (self.bnum) * self.w
            h_end = h_sta + self.h
            w_end = w_sta + self.w
            for img_id, (img) in enumerate(initdata):
                img = img.permute(1, 2, 0).contiguous().cpu().detach().numpy()  # H W C
                tdata = img[h_sta:h_end, w_sta:w_end, :]
                tdata = tdata.reshape((-1, C))
                if ((img_id + 1) % 10 == 0):
                    print(img_id)
                if img_id == 0:
                    data = tdata
                else:
                    data = np.append(data, tdata, axis=0)
            data = np.float32(data)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1)
            flags = cv2.KMEANS_RANDOM_CENTERS
            n_sta = block_id * self.bnod
            n_end = n_sta + self.bnod
            a, label, anchor[n_sta:n_end, :] = cv2.kmeans(data, self.bnod, None, criteria, 10, flags)
            for i in range(self.bnod):
                ind = (label == i).flatten()
                sigma[n_sta+i, :] = data[ind, :].var(axis=0)
        self.anchor = nn.Parameter(torch.from_numpy(anchor))
        self.sigma = nn.Parameter(torch.from_numpy(sigma))
        return


class GraphProcess(nn.Module):
    def __init__(self, bnum, bnod, dim, loop):
        super(GraphProcess, self).__init__()
        self.loop = loop
        self.bnum = bnum
        self.bnod = bnod
        self.dim = dim
        self.node_num = bnum * bnum * bnod
        self.proj = GraphNet(self.bnum, self.bnod, self.dim)

    def reproj(self,Q,Z):
        self.B, self.Dim, _ = Z.size()
        res=torch.zeros([self.B,self.Dim,self.H,self.W]).cuda(0)
        for node_id in range(self.node_num):
            block_id=math.floor(node_id/self.bnod)
            h_sta=math.floor(block_id/self.bnum)*self.h
            w_sta=block_id%(self.bnum)*self.w
            h_end=h_sta+self.h
            w_end=w_sta+self.w
            res[:,:, h_sta:h_end , w_sta:w_end]+=torch.matmul(Z[:,:,node_id].unsqueeze(2),Q[:,node_id,:].unsqueeze(1)).view(self.B,self.Dim,self.h,self.w)
        return res

    def forward(self, x, Q, g):  #
        _, _, self.H, self.W = x.size()
        self.h = math.floor(self.H / self.bnum)
        self.w = math.floor(self.W / self.bnum)
        rg = self.reproj(Q, g)
        return rg


class graph_block_d(nn.Module):
    def __init__(self, bnum, bnod, dim, loop, bias, k):
        super(graph_block_d, self).__init__()
        self.gp = GraphProcess(bnum, bnod, dim, loop)  # 64
        self.gconv = ParallGCNetPyg(dim, loop, bnum, k, dims=128)

    def forward(self, x):
        g, A, Q = self.gp.proj(x)  # x is BCHW
        xg = self.gconv(g, A)
        xG = self.gp.forward(x, Q, xg) + x  # + img3
        return xG


class SpatialAttention(nn.Module):
    def __init__(self, channels):
        super(SpatialAttention, self).__init__()
        self.conv = BasicConv2d(channels, 1, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, xin):
        att = self.conv(xin)
        return self.sigmoid(att)


class ChannelAttention(nn.Module):
    def __init__(self, ch, ratio=16, flag_AE_CA='0'):
        super(ChannelAttention, self).__init__()
        self.p = nn.ParameterList([nn.init.uniform_(nn.Parameter(torch.empty(ch, 1, 1)))]).cuda()
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # input is BCHW, output is BC11
        self.conv_re = BasicConv2d(ch, ch, kernel_size=3, padding=1)
        self.max_fc = BasicConv2d(ch, ch, kernel_size=1)
        self.sig = nn.Sigmoid()
        self.flag_AE_CA = flag_AE_CA

    def forward(self, xin):
        max = self.max_pool(xin)
        xin = xin + self.p[0]*(self.conv_re(max - xin)) * xin
        CA = self.max_fc(self.max_pool(xin))
        xout = CA * xin + xin
        return xout


class AE(nn.Module):
    def __init__(self, ch, flag_AE='c55' +'AE1'):  #
        super(AE, self).__init__()
        self.SA = SpatialAttention(ch)
        self.CA = ChannelAttention(ch,flag_AE_CA=flag_AE)

    def forward(self, inx):
        x = self.CA(inx)
        out = self.SA(x) * x + x  # inx
        return out


class CalculateUnitAE(nn.Module):
    def __init__(self, in_Channel1=0, in_Channel2=0, in_Channel3=0, out_Channel=0, SA_Channel=0, flag='c55'):
        super(CalculateUnitAE, self).__init__()
        self.sig = nn.Sigmoid()
        self.AE1 = AE(in_Channel1, flag_AE=flag +'AE1')
        self.AE2 = AE(in_Channel2, flag_AE=flag +'AE2')
        self.AE3 = AE(in_Channel3, flag_AE=flag +'AE3')
        if (SA_Channel != 0):
            self.convSA = BasicConv2d(SA_Channel, 1, 3, padding=1)
        self.conv = nn.Sequential(BasicConv2d(in_Channel1 + in_Channel2 + in_Channel3, out_Channel, 3, padding=1), BasicConv2d(out_Channel, out_Channel, 3, padding=1))

    def forward(self, in1, in2=0, in3=0, inSA=0):
        if (torch.is_tensor(inSA)):
            SA = self.sig(self.convSA(inSA))
        else:
            SA = 0
        x = self.AE1(in1) * (SA + 1)
        if (torch.is_tensor(in2)):
            in2 = self.AE2(in2) * (SA + 1)
            x = torch.cat([x, in2], dim=1)
        if (torch.is_tensor(in3)):
            in3 = self.AE3(in3) * (SA + 1)
            x = torch.cat([x, in3], dim=1)
        out = self.conv(x)
        return out


class decoder(nn.Module):
    def __init__(self):
        super(decoder, self).__init__()

        a5_channel = 256
        a4_channel = 256
        a3_channel = 256
        a2_channel = 128
        a1_channel = 64

        self.up16 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)
        self.up8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.up4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dw2 = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)

        self.c55 = CalculateUnitAE(a5_channel, a5_channel, a4_channel, a4_channel, 0, flag='c55')
        self.c54 = CalculateUnitAE(a4_channel, a4_channel, a3_channel, a3_channel, a4_channel, flag='c54')
        self.c53 = CalculateUnitAE(a3_channel, a3_channel, a2_channel, a2_channel, a3_channel, flag='c53')
        self.c52 = CalculateUnitAE(a2_channel, a2_channel, a1_channel, a1_channel, a2_channel, flag='c52')
        self.c51 = CalculateUnitAE(a1_channel, a1_channel, a1_channel, a1_channel, a1_channel, flag='c51')

        self.c45 = CalculateUnitAE(a4_channel, a4_channel, a3_channel, a3_channel, a4_channel, flag='c45')
        self.c44 = CalculateUnitAE(a3_channel, a2_channel, 0, a2_channel, a3_channel, flag='c44')
        self.c43 = CalculateUnitAE(a2_channel, a1_channel, 0, a1_channel, a2_channel, flag='c43')
        self.c42 = CalculateUnitAE(a1_channel, a1_channel, 0, a1_channel, a1_channel, flag='c42')

        self.c35 = CalculateUnitAE(a3_channel, a3_channel, a2_channel, a2_channel, a3_channel, flag='c35')
        self.c34 = CalculateUnitAE(a2_channel, a1_channel, 0, a1_channel, a2_channel, flag='c34')
        self.c33 = CalculateUnitAE(a1_channel, a1_channel, 0, a1_channel, a1_channel, flag='c33')

        self.c25 = CalculateUnitAE(a2_channel, a2_channel, a1_channel, a1_channel, a2_channel, flag='c25')
        self.c24 = CalculateUnitAE(a1_channel, a1_channel, 0, a1_channel, a1_channel, flag='c24')

        self.c15 = CalculateUnitAE(a1_channel, a1_channel, 0, a1_channel, a1_channel, flag='c15')

        self.S5 = nn.Conv2d(a4_channel, 1, 3, stride=1, padding=1)
        self.S4 = nn.Conv2d(a3_channel, 1, 3, stride=1, padding=1)
        self.S3 = nn.Conv2d(a2_channel, 1, 3, stride=1, padding=1)
        self.S2 = nn.Conv2d(a1_channel, 1, 3, stride=1, padding=1)
        self.S1 = nn.Conv2d(a1_channel, 1, 3, stride=1, padding=1)

        self.R2 = nn.Conv2d(a1_channel, 1, 3, stride=1, padding=1)
        self.R3 = nn.Conv2d(a1_channel, 1, 3, stride=1, padding=1)
        self.R4 = nn.Conv2d(a1_channel, 1, 3, stride=1, padding=1)
        self.R5 = nn.Conv2d(a1_channel, 1, 3, stride=1, padding=1)

        self.sig = nn.Sigmoid()

    def forward(self, xC5, xC4, xC3, xC2, xC1, xG5, xG4, xG3, xG2, xG1):
        x55 = self.c55(xC5, xG5, self.dw2(xG4), 0)
        x45 = self.c45(self.up2(x55), xG4, self.dw2(xG3), self.up2(x55))
        x35 = self.c35(self.up2(x45), xG3, self.dw2(xG2), self.up2(x45))
        x25 = self.c25(self.up2(x35), xG2, self.dw2(xG1), self.up2(x35))
        x15 = self.c15(self.up2(x25), xG1, 0, self.up2(x25))

        x54 = self.c54(xC4, self.up2(x55), x45, self.up2(x55))
        x44 = self.c44(self.up2(x45), x35, 0, self.up2(x45))
        x34 = self.c34(self.up2(x35), x25, 0, self.up2(x35))
        x24 = self.c24(self.up2(x25), x15, 0, self.up2(x25))

        x53 = self.c53(xC3, self.up2(x54), x44, self.up2(x54))
        x43 = self.c43(self.up2(x44), x34, 0, self.up2(x44))
        x33 = self.c33(self.up2(x34), x24, 0, self.up2(x34))

        x52 = self.c52(xC2, self.up2(x53), x43, self.up2(x53))
        x42 = self.c42(self.up2(x43), x33, 0, self.up2(x43))

        x51 = self.c51(xC1, self.up2(x52), x42, self.up2(x52))

        # The output of the upper edge line of the triangle.
        s1 = self.S1(x51)
        s2 = self.S2(x52)
        s3 = self.S3(x53)
        s4 = self.S4(x54)
        s5 = self.S5(x55)

        # The lateral output of the diagonal line in a triangle.
        r2 = self.R2(x42)
        r3 = self.R3(x33)
        r4 = self.R4(x24)
        r5 = self.R5(x15)

        return s1, self.up2(s2), self.up4(s3), self.up8(s4), self.up16(s5), r2, r3, r4, r5

# CMNFNet
class CMNFNet(nn.Module):
    def __init__(self,bnum, bnod, dim,loop,bias,init, k, res_pretrained):
        super(CMNFNet,self).__init__()
        self.dim = dim
        self.upsample16 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)
        self.upsample8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.sig = nn.Sigmoid()
        # VGG backbone
        self.vgg = vgg.VGG('rgb')
        self.layer1 = self.vgg.conv1
        self.layer2 = self.vgg.conv2
        self.layer3 = self.vgg.conv3
        self.layer4 = self.vgg.conv4
        self.layer5 = self.vgg.conv5
        # ################################################################################################################################
        self.pos_embed3 = nn.Parameter(torch.zeros(1, 256, 64, 64))
        nn.init.trunc_normal_(self.pos_embed3, std=.02)
        self.pos_embed4 = nn.Parameter(torch.zeros(1, 256, 32, 32))
        nn.init.trunc_normal_(self.pos_embed4, std=.02)
        self.pos_embed5 = nn.Parameter(torch.zeros(1, 256, 16, 16))
        nn.init.trunc_normal_(self.pos_embed5, std=.02)

        c1, c2, c3, c4, c5 = 64, 128, 256, 512, 512
        self.conv4 = nn.Sequential(BasicConv2d(c4, c3, kernel_size=3, padding=1),
                                   BasicConv2d(c3, c3, kernel_size=1, padding=0))  # 512-256

        self.conv5 = nn.Sequential(BasicConv2d(c5, c3, kernel_size=3, padding=1),
                                   BasicConv2d(c3, c3, kernel_size=1, padding=0))  # 512-256

        # GCN backbone
        self.convg1 = nn.Sequential(BasicConv2d(3, 2 * dim, kernel_size=3, stride=1, padding=1), BasicConv2d(2*dim, 2*dim, kernel_size=1, stride=1, padding=0))  # channel:3-64, size:256

        self.convg2 = nn.Sequential(BasicConv2d(2*dim, 4*dim, kernel_size=3, stride=2, padding=1), BasicConv2d(4*dim, 4*dim, kernel_size=1, stride=1, padding=0)) # channel: 64-128, size: 256→128

        self.convg3 = nn.Sequential(BasicConv2d(4 * dim, 8 * dim, kernel_size=3, stride=2, padding=1), BasicConv2d(8*dim, 8*dim, kernel_size=1, stride=1, padding=0))  # channel: 128-256, size: 128→64
        self.GCNblock3 = graph_block_d(int(bnum), int(bnod), 8 * dim, int(loop), bias, k)

        self.convg4 = nn.Sequential(BasicConv2d(8 * dim, 8 * dim, kernel_size=3, stride=2, padding=1), BasicConv2d(8*dim, 8*dim, kernel_size=1, stride=1, padding=0))  # channel: 256-256, size: 64→32
        self.GCNblock4 = graph_block_d(int(bnum), int(bnod), 8 * dim, int(loop), bias, k)

        self.convg5 = nn.Sequential(BasicConv2d(8 * dim, 8 * dim, kernel_size=3, stride=2, padding=1), BasicConv2d(8*dim, 8*dim, kernel_size=1, stride=1, padding=0))  # channel: 256-256, size: 32→16
        self.GCNblock5 = graph_block_d(int(bnum), int(bnod), 8 * dim, int(loop), bias, k)
        ###############################################
        self.decoder = decoder()

    def forward(self, Xin):
        #########################################
        xC1 = self.layer1(Xin)  # 1/2, 64
        xG1 = self.convg1(Xin)
        #########################################
        xC2 = self.layer2(xC1)  # 1/4, 128
        xG2 = self.convg2(xG1)
        #########################################
        xC3 = self.layer3(xC2)  # 1/8, 256
        xg3 = self.convg3(xG2)
        xG3 = self.GCNblock3(xg3) + self.pos_embed3
        #########################################
        xc4 = self.layer4(xC3)  # 1/8, 512
        xC4 = self.conv4(xc4)  # 1/8, 256
        xg4 = self.convg4(xG3)
        xG4 = self.GCNblock4(xg4) + self.pos_embed4
        #########################################
        xc5 = self.layer5(xc4)  # 1/16, 512
        xC5 = self.conv5(xc5)  # 1/16, 256
        xg5 = self.convg5(xG4)
        xG5 = self.GCNblock5(xg5) + self.pos_embed5

        s1, s2, s3, s4, s5, s6, s7, s8, s9 = self.decoder(xC5, xC4, xC3, xC2, xC1, xG5, xG4, xG3, xG2, xG1)

        s1 = self.sig(s1.squeeze(dim=1))
        s2 = self.sig(s2.squeeze(dim=1))
        s3 = self.sig(s3.squeeze(dim=1))
        s4 = self.sig(s4.squeeze(dim=1))
        s5 = self.sig(s5.squeeze(dim=1))
        s6 = self.sig(s6.squeeze(dim=1))
        s7 = self.sig(s7.squeeze(dim=1))
        s8 = self.sig(s8.squeeze(dim=1))
        s9 = self.sig(s9.squeeze(dim=1))

        return s1, s2, s3, s4, s5, s6, s7, s8, s9