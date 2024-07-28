import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4,5,6,7' 
# os.environ['export']=' DISPLAY=127.0.0.1:2.0'
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import torch
import time
from utilities import *
import torch.multiprocessing as mp
import libtorch_staggerfd_cuda
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--node_rank", type=int)
parser.add_argument("--num_proc", type=int)
parser.add_argument("--sub_forward", type=int)
parser.add_argument("--world_size", type=int)
parser.add_argument("--forward_num", type=int)
parser.add_argument("--master_addr", default="127.0.0.1", type=str)
parser.add_argument("--master_port", default="12355", type=str)

args = parser.parse_args()
import torch
#import visdom
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torch import nn, optim

import scipy.io as sio

class NN(nn.Module):
    def __init__(self,x, num_input, w0, h0, vpmax, vpmin, vsmax, vsmin,vp0,vs0,denm0,denm2):
        super(NN, self).__init__()
        self.num_input = num_input
        self.w0 = w0
        self.h0 = h0
        self.vpmax = vpmax
        self.vpmin = vpmin
        self.vsmax = vsmax
        self.vsmin = vsmin
        self.vvpmax = vpmax-vpmin
        self.vvpmin = vpmin-vpmax
        self.vvsmax = vsmax-vsmin
        self.vvsmin = vsmin-vsmax
        self.vp0 = vp0
        self.vs0 = vs0
        self.denm0 = denm0
        self.denm2 = denm2
        self.maxlinep = vpmax*torch.ones_like(vp0)
        self.minlinep = vpmin*torch.ones_like(vp0)
        self.maxlines = vsmax*torch.ones_like(vs0)
        self.minlines = vsmin*torch.ones_like(vs0)
        self.x = x
        self.vpnet1 = nn.Sequential(
            nn.Linear(num_input, w0*h0*8,bias=False),        
            nn.ReLU(),
        )
    
        self.vpnet2 = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(8, 128, (4, 4), stride=(1, 1), padding="same", bias=False),
            nn.LeakyReLU(negative_slope=0.01),
            
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(128, 64, (4, 4), stride=(1, 1), padding="same", bias=False),
            nn.LeakyReLU(negative_slope=0.01),
            
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(64, 32, (4, 4), stride=(1, 1), padding="same", bias=False),
            nn.LeakyReLU(negative_slope=0.01),
            
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(32, 16, (4, 4), stride=(1, 1), padding="same", bias=False),
            nn.LeakyReLU(negative_slope=0.01),
            
#             nn.UpsamplingBilinear2d(scale_factor=2),
#             nn.Conv2d(16, 8, (4, 4), stride=(1, 1), padding="same", bias=False),
#             nn.LeakyReLU(negative_slope=0.01),
            
            nn.Conv2d(16, 1, (4, 4), stride=(1, 1), padding="same", bias=False),
            
        )
        self.vsnet1 = nn.Sequential(
            nn.Linear(num_input, w0*h0*8,bias=False),        
            nn.ReLU(),
        )
    
        self.vsnet2 = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(8, 128, (4, 4), stride=(1, 1), padding="same", bias=False),
            nn.LeakyReLU(negative_slope=0.01),
            
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(128, 64, (4, 4), stride=(1, 1), padding="same", bias=False),
            nn.LeakyReLU(negative_slope=0.01),
            
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(64, 32, (4, 4), stride=(1, 1), padding="same", bias=False),
            nn.LeakyReLU(negative_slope=0.01),
            
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(32, 16, (4, 4), stride=(1, 1), padding="same", bias=False),
            nn.LeakyReLU(negative_slope=0.01),
            
#             nn.UpsamplingBilinear2d(scale_factor=2),
#             nn.Conv2d(16, 8, (4, 4), stride=(1, 1), padding="same", bias=False),
#             nn.LeakyReLU(negative_slope=0.01),
            
            nn.Conv2d(16, 1, (4, 4), stride=(1, 1), padding="same", bias=False),
            
        )
    def forward(self, sx, rank, input_vector, temp, s,den,device):
        x = self.x
        vp = self.vpnet1(x)
        vp = vp.reshape(-1, int(self.num_input), int(self.w0), int(self.h0))
        vp = self.vpnet2(vp)
        # print('vp',vp)
        vs = self.vsnet1(x)
        vs = vs.reshape(-1, int(self.num_input), int(self.w0), int(self.h0))
        vs = self.vsnet2(vs)

        vp = ((self.vvpmax - self.vvpmin) * torch.tanh(vp) + (self.vvpmax + self.vvpmin))/2.0
        vp = torch.squeeze(vp)
        y_shape0 = (vp.shape[0] - self.vp0.shape[0])//2
        y_shape1 = (vp.shape[1] - self.vp0.shape[1])//2
        dk_vp = vp[y_shape0:y_shape0+self.vp0.shape[0],y_shape1:y_shape1+self.vp0.shape[1]]
        # v_mean_p=torch.sqrt(torch.sum(self.vp0**2))
        # g_mean_p=torch.sqrt(torch.sum(dk_vp**2))
        # alpha_p=v_mean_p/g_mean_p  
        # velocitvp = self.vp0 + alpha_p * dk_vp * pur
        velocitvp = self.vp0 + dk_vp
        velocitvp = torch.where(velocitvp > self.vpmax, self.maxlinep, velocitvp)
        velocitvp = torch.where(velocitvp < self.vpmin, self.minlinep, velocitvp)
                
        vs = ((self.vvsmax - self.vvsmin) * torch.tanh(vs) + (self.vvsmax + self.vvsmin))/2.0
        vs = torch.squeeze(vs)
        dk_vs = vs[y_shape0:y_shape0+self.vp0.shape[0],y_shape1:y_shape1+self.vp0.shape[1]]
        # v_mean_s=torch.sqrt(torch.sum(self.vs0**2))
        # g_mean_s=torch.sqrt(torch.sum(dk_vs**2))
        # alpha_s=v_mean_s/g_mean_s  
        # velocitvs = self.vs0 + alpha_s * dk_vs * pur
        velocitvs = self.vs0 + dk_vs
        velocitvs = torch.where(velocitvs > self.vsmax, self.maxlines, velocitvs)
        velocitvs = torch.where(velocitvs < self.vsmin, self.minlines, velocitvs)
        
        velocitvs = torch.where((velocitvp/velocitvs)<1.414,velocitvp/1.414,velocitvs)
        if rank==0:
            imagesc(velocitvp)
            plt.savefig('./vp_iter/vp' + str(0) + '.png')
            imagesc(velocitvs)
            plt.savefig('./vp_iter/vs' + str(0) + '.png')
            imagesc(vp)
            plt.savefig('./vp_iter/dkvp' + str(0) + '.png')
            imagesc(vs)
            plt.savefig('./vp_iter/dkvs' + str(0) + '.png')
            torch.save({'velocitvp':velocitvp.detach().cpu().clone(),'velocitvs':velocitvs.detach().cpu().clone()},'./res2.pt')
        input_vector[6] = int(sx[rank])
        # print('model:',rank,input_vector[6])
        nbc = int(input_vector[16].clone())
        pad_top = int(input_vector[17].clone())
        dtx = input_vector[3].clone()
        vp1 = pad(velocitvp,nbc,pad_top)
        vs1 = pad(velocitvs,nbc,pad_top)
        ca=torch.mul(vp1**2,den)
        cm=torch.mul((vs1**2),den)
        cl=ca - 2*cm
        den1 = den
        cm1=cm
        cl = cl * self.denm0
        cam = torch.zeros_like(ca, device=torch.device('cuda', device))
        cam[pad_top-1,:] = 2 * cm[pad_top-1,:] - ca[pad_top-1,:]
        ca = ca + cam
        b=dtx*torch.reciprocal(den)
        b = b * self.denm2
        b1=b
        [uu1,seismo_v_d,illum_div1]=staggeredfd_py_cuda(input_vector,temp,ca,cl,cm,cm1,b,b1,s,device)
        # imagesc(seismo_v_d)
        # plt.savefig('./vp_iter/obsdata' + str(rank) + '.png')
        return uu1,seismo_v_d,illum_div1 
        # [uu1,seismo_v_d,illum_div1]=staggeredfd_py_cuda(input_vector,temp,ca,cl,cm,cm1,b,b1,s)
    
class fd2d(torch.nn.Module):
    def __init__(self,vp0,vs0,denm0,denm2):
        super(fd2d, self).__init__()
#         self.params = nn.ParameterDict({'vp': nn.Parameter(torch.empty(output_features, input_features))})
        self.params1 = torch.nn.Parameter(torch.empty(26, 120))
        # self.params2 = torch.nn.Parameter(torch.empty(26, 120))
        self.params2 = torch.empty(26, 120)
        self.params1.data = vp0.data
        self.params2.data = vs0.data
        self.denm0 = denm0
        self.denm2 = denm2

#                                               torch.nn.Parameter(vs0,requires_grad=True)])
#         self.params[0]=vp0
    def forward(self, sx, rank, input_vector, temp, s,den,device):
        input_vector[6] = int(sx[rank])
        # print('model:',rank,input_vector[6])
        nbc = int(input_vector[16].clone())
        pad_top = int(input_vector[17].clone())
        dtx = input_vector[3].clone()
        vp1 = pad(self.params1,nbc,pad_top)
        vs1 = pad(self.params2,nbc,pad_top)
        ca=torch.mul(vp1**2,den)
        cm=torch.mul((vs1**2),den)
        cl=ca - 2*cm
        den1 = den
        cm1=cm
        cl = cl * self.denm0
        cam = torch.zeros_like(ca, device=torch.device('cuda', device))
        cam[pad_top-1,:] = 2 * cm[pad_top-1,:] - ca[pad_top-1,:]
        ca = ca + cam
        b=dtx*torch.reciprocal(den)
        b = b * self.denm2
        b1=b
        [uu1,seismo_v_d,illum_div1]=staggeredfd_py_cuda(input_vector,temp,ca,cl,cm,cm1,b,b1,s,device)
        return uu1,seismo_v_d,illum_div1 
        # [uu1,seismo_v_d,illum_div1]=staggeredfd_py_cuda(input_vector,temp,ca,cl,cm,cm1,b,b1,s)

def cal_Grad2(local_rank, node_rank, local_size, world_size,sub_forward,forward_num,input_vector,denm0,denm2,\
              temp,s,sx,vp0,vs0,seismo_v_d,x,vpmax,vpmin,vsmax,vsmin,taper,taper1):
    nbc = int(input_vector[16].clone())
    pad_top = int(input_vector[17].clone())
    ng = int(input_vector[4].clone())
    den = pad(torch.ones_like((vp0)),nbc,pad_top)
    # create default process group
    rank = local_rank + node_rank * local_size
    shot_i = (local_rank + 1) * forward_num - forward_num + node_rank
    print(rank,local_rank,node_rank,local_size,shot_i)
    torch.cuda.set_device(local_rank)
    dist.init_process_group("gloo",
                            init_method="tcp://{}:{}".format(args.master_addr, args.master_port),
                            rank=rank,
                            world_size=world_size)
    input_vector[6] = int(sx[rank])
    
    # device_id = rank
    # print(device_id,rank)
    denm0 = denm0.to(local_rank)
    denm2 = denm2.to(local_rank)
    vp0 = vp0.to(local_rank)
    vs0 = vs0.to(local_rank)   
    num_layer=4
    num_input = 8
    w0=int(np.ceil(vp0.shape[0]/(2**num_layer)))
    h0=int(np.ceil(vp0.shape[1]/(2**num_layer)))
    x = x.to(local_rank)
    # print('randx',x)
    model = NN(x, num_input, w0, h0, vpmax, vpmin, vsmax, vsmin,vp0,vs0,denm0,denm2).to(local_rank)
    # model = fd2d(vp0.clone(),vs0.clone(),denm0.clone(),denm2.clone()).to(local_rank)
    ddp_model = DDP(model, device_ids=[local_rank], output_device=local_rank) 
    optimizer = torch.optim.Adam(ddp_model.parameters(),lr=2e-4)
    # optimizer = torch.optim.Adam(ddp_model.parameters(),lr=10)
    seismo_v_d1 = seismo_v_d[rank,:,:].to(local_rank)
    taper = taper.to(local_rank)
    taper1 = taper1.to(local_rank)
    input_vector = input_vector.to(local_rank)
    temp = temp.to(local_rank)
    den = den.to(local_rank)
    s = s.to(local_rank)
    for i in range(2000):
        T0=time.time()
        optimizer.zero_grad()
        for j in range(sub_forward):
            # print(i,rank)
            [seismo_u,seismo_w,illum_div1] = ddp_model(sx,rank,input_vector,temp,s,den,local_rank)
            seismo_ww = torch.zeros_like(seismo_w,device=local_rank)
#             seismo_w = bp_filter(seismo_w,8e-5,1,1,40,40+5)
#             seismo_w = seismo_w * taper[rank,:,:] * taper1[rank,:,:] 
            for ii in range(ng):
                seismo_ww[:,ii] = seismo_w[:,ii]/torch.max(torch.abs(seismo_w[:,ii]))  
#             seismo_ww = seismo_ww * taper[rank,:,:] * taper1[rank,:,:] 
            tracesum1 = torch.sum(seismo_v_d1**2)
            tracesum2 = torch.sum(seismo_ww**2)
            loss1 = (1.0 - torch.abs(torch.sum(seismo_ww*seismo_v_d1/(torch.sqrt(tracesum1)*torch.sqrt(tracesum2)))))**2

            loss2 = torch.sum((seismo_ww-seismo_v_d1)**2)#/(torch.sqrt(tracesum1)*torch.sqrt(tracesum2))
            loss = loss2
            loss.backward()
            if rank==0:
                imagesc(seismo_ww)
                plt.savefig('./vp_iter/seismo_ww' + str(0) + '.png')
                imagesc(seismo_v_d1)
                plt.savefig('./vp_iter/seismo_v_d1' + str(0) + '.png')
            # for p,para in enumerate(ddp_model.parameters()):
            #     if p==0:
            #         para.data.clamp_(1000, 2500)
            #         para.grad = para.grad / illum_div1[2:-40,40:-40]
            #     if p==1:
            #         para.data.clamp_(1000/1.732, 2500/1.732)
            #         para.grad = para.grad / illum_div1[2:-40,40:-40]
        optimizer.step()
        T1=time.time()
        if rank==0:
            print(loss,loss1,loss2,i,'time in shot 1:',T1-T0)
        # if rank==0:
        #     for p,para in enumerate(ddp_model.parameters()):
        #         if p==0:
        #             velocityp = para.detach().clone()
        #             imagesc(velocityp)
        #             plt.savefig('./vp_iter/vp' + str(i) + '.png')
        #             velocityp = para.grad.detach().clone()
        #             imagesc(velocityp)
        #             plt.savefig('./dk_vp_iter/vp' + str(i) + '.png')
        #         elif p==1:
        #             velocitys = para.detach().clone()
        #             imagesc(velocitys)
        #             plt.savefig('./vs_iter/vs' + str(i) + '.png')
        #             velocitys = para.grad.detach().clone()
        #             imagesc(velocitys)
        #             plt.savefig('./dk_vs_iter/vs' + str(i) + '.png')

            # sio.savemat('./restemp/NNwadi_04_11_' + str(i) + '.mat',{'Wadivp04_11':velocityp.detach().cpu().numpy(),'Wadivs04_11':velocitys.detach().cpu().numpy()})

if __name__ == '__main__':
    dicttemp = torch.load('./modeltest3.pt')
    for key, value in dicttemp.items():
        exec(key + '=dicttemp["' + key + '"].to("cpu")') 
    local_size = torch.cuda.device_count()
    print("local_size: %s" % local_size)
    x = torch.tensor([[0.5811, 0.6117, 0.0138, 0.3378, 0.0043, 0.8173, 0.1347, 0.1372]])
    mp.spawn(cal_Grad2,
        args=(args.node_rank, local_size, args.world_size,args.sub_forward,args.forward_num,input_vector,\
              denm0,denm2,temp,s,sx,vp0,vs0,seismo_v_d,x,vpmax,vpmin,vsmax,vsmin,torch.tensor([[1]]),torch.tensor([[1]]),),
        nprocs=args.num_proc,
        join=True)



 
