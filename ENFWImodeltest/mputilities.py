import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7' 
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

class fd2d(torch.nn.Module):
    def __init__(self,vp0,vs0):
        super(fd2d, self).__init__()
#         self.params = nn.ParameterDict({'vp': nn.Parameter(torch.empty(output_features, input_features))})
        self.params1 = torch.nn.Parameter(torch.empty(26, 120))
        self.params2 = torch.nn.Parameter(torch.empty(26, 120))
        self.params1.data = vp0.data
        self.params2.data = vs0.data

#                                               torch.nn.Parameter(vs0,requires_grad=True)])
#         self.params[0]=vp0
    def forward(self, input_vector,temp,s,den,device):
        vp1 = pad(self.params1,40,2)
        vs1 = pad(self.params2,40,2)
        ca=torch.mul(vp1**2,den)
        cm=torch.mul((vs1**2),den)
        cl=ca - 2*cm
        cm1=cm
        dtx = input_vector[3]
        b=dtx*torch.reciprocal(den)
        b1=b
        [uu1,seismo_v_d,illum_div1]=staggeredfd_py_cuda(input_vector,temp,ca,cl,cm,cm1,b,b1,s,device)
        return uu1,seismo_v_d,illum_div1 
        # [uu1,seismo_v_d,illum_div1]=staggeredfd_py_cuda(input_vector,temp,ca,cl,cm,cm1,b,b1,s)


def obs_fd2d(device,i,vp1,vs1,den,input_vector,sx,temp,s,returnQue,p2c):
    T1=time.time()

    print(i)
    pad_top=2
    cl=torch.mul(vp1**2-2*vs1**2,den)
    cm=torch.mul((vs1**2),den)
    ca=cl + 2*cm
    den = den
    cm1=cm
    cl[pad_top,:] = 0
    ca[pad_top,:] = cm[pad_top,:]
    b=input_vector[3]*torch.reciprocal(den)
    b[pad_top,:] = 2 * b[pad_top,:]
    b1=b
    print(sx[i])
    input_vector[6] = int(sx[i])
    input_vector=input_vector.cuda(device)
    temp=temp.cuda(device)
    ca=ca.cuda(device)
    cl=cl.cuda(device)
    cm=cm.cuda(device)
    cm1=cm1.cuda(device)
    b=b.cuda(device)
    b1=b1.cuda(device)
    s=s.cuda(device)
    # for le in ['input_vector','temp','ca','cl','cm','cm1','b','b1','s']:
    #     print(le)
    #     exec(le + '=' + le + '.to("cuda:0")')
    #     exec(le + '=' + le + '.to("cuda:' + str(device) + '")')
    [seismo_u,seismo_w,illum_div]=staggeredfd_py_cuda(input_vector,temp,ca,cl,cm,cm1,b,b1,s,device)

    # [seismo_u,seismo_w,illum_div]=libtorch_staggerfd_cuda.forward(input_vector,temp,ca,cl,cm,cm1,b,b1,s,device)
    imagesc(seismo_w)
    plt.savefig('./dk_vp_iter/vp' + str(i) + '.png')
    # [seismo_u,seismo_w,illum_div]=staggeredfd_py_cuda(input_vector,temp,ca,cl,cm,cm1,b,b1,s,device)
    print(torch.sum(seismo_u))
    T2=time.time()
    returnQue.put((seismo_w.clone(),i))
    p2c.get()
    print('Time used in shot:' + str(i) + ' = ' + str(T2-T1))

def grad_fd2d(device,i,seismo_v_d,vp1,vs1,den,input_vector,sx,temp,s,returnQue,p2c):
    net = fd2d(vp1.clone(),vs1.clone())
    net = net.cuda(device)
    s=s.cuda(device)
    T1=time.time()
    input_vector[6] = int(sx[i])
    [seismo_u,seismo_w,illum_div1] = net(input_vector,temp,s,den)
    loss = torch.sum((seismo_w-seismo_v_d[i,:,:])**2)
    res[i] = res[i] + loss.data
    loss.backward()
    returnQue.put((loss.clone(),i))
    p2c.get()
    print('Time used in shot:' + str(i) + ' = ' + str(T2-T1))
    T2=time.time()

def Generate_obs_data(n_proc,device_num,vp0,vs0,input_vector,sx,temp,s):
    nbc = int(input_vector[16].clone())
    pad_top = int(input_vector[17].clone())
    den = pad(torch.ones_like(vp0),nbc,pad_top)
    vp1 = pad(vp0,nbc,pad_top)
    vs1 = pad(vs0,nbc,pad_top)
    T1=time.time()
    nt = int(input_vector[0])
    # ns = int(len(sx))
    ns = n_proc
    ng = int(input_vector[4])
    seismo_v_d = torch.zeros((ns,nt,ng)).to('cuda')
    ctx = mp.get_context('spawn')
    returnQue = ctx.SimpleQueue()
    p2c = ctx.SimpleQueue()
    pool = []
    print(ns)
    for i in range(ns):
        divied_target=math.floor(i/n_proc*device_num)
        process = ctx.Process(target=obs_fd2d, args=(divied_target,i,vp1,vs1,den,input_vector,sx,temp,s,returnQue,p2c)) 
        process.start()
        pool.append(process)
    for process in pool:
        res = returnQue.get()  
        seismo_v_d[res[1],:,:] = res[0]
        # print(res)
        del res
        p2c.put(0) # wait for until child process is ready
    # for process in pool:
    #     process.join()
    T2=time.time()
    # imagesc(seismo_v_d[0,:,:])
    # plt.savefig('./temp.png')
    print('Generate obs data. Time used all shot = ' + str(T2-T1))
    return seismo_v_d    

def cal_Grad(n_proc,device_num,vp0,vs0,input_vector,sx,temp,s,nbc,pad_top,seismo_v_d):
    ctx = mp.get_context('spawn')
    returnQue = ctx.SimpleQueue()
    p2c = ctx.SimpleQueue()
    pool = []
    for i in range(1):
        divied_target=math.floor(i/n_proc*device_num)
        process = ctx.Process(target=grad_fd2d, args=(divied_target,i,seismo_v_d,vp1,vs1,den,input_vector,sx,temp,s,returnQue,p2c)) 
        process.start()
        pool.append(process)
    for process in pool:
        res = returnQue.get()  
        seismo_v_d[res[1],:,:] = res[0]
        # print(res)
        del res
        p2c.put(0) # wait for until child process is ready

    T2=time.time()
    optimizer.step()# This will update the shared parameters
    optimizer.zero_grad()   

def cal_Grad1(rank,world_size,input_vector,temp,s,sx,vp0,vs0,seismo_v_d):
    print(1,rank)
    den = pad(torch.ones_like((vp0)),40,2)
    # create default process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    # create local model
    input_vector[6] = int(sx[rank])
    device_id = rank
    # device_id = rank
    # print(device_id,rank)
    model = fd2d(vp0.clone(),vs0.clone()).to(device_id)
    ddp_model = DDP(model, device_ids=[device_id])
    optimizer = torch.optim.Adam(ddp_model.parameters(),lr=10)
    for i in range(20):
        optimizer.zero_grad()
        input_vector = input_vector.to(device_id)
        temp = temp.to(device_id)
        s = s.to(device_id)
        den = den.to(device_id)
        seismo_v_d1 = seismo_v_d[rank,:,:].to(device_id)
        print(i)
        [seismo_u,seismo_w,illum_div1] = ddp_model(input_vector,temp,s,den,device_id)
        loss = torch.sum((seismo_w-seismo_v_d1)**2)
        loss.backward()
        print(loss,i)
        optimizer.step()

   