import torch
# import taichi
import numpy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import math
def norm_trace(seis):
    data_out = np.zeros(np.shape(seis))
    for k in range(np.size(seis,axis=1)):
        data_out[:,k] = seis[:,k]/np.max(np.abs(seis[:,k]))
    seis=data_out
    return seis
def envelope(xr):
    for i in range(np.size(xr,axis=1)):
        hx = fftpack.hilbert(xr[:,i])
        xr[:,i] = np.sqrt(np.power(xr[:,i],2) + np.power(hx,2))
    return xr
def imagesc(damp,alpha=1,cmin=1,cmax=1):
    damp = damp.detach().cpu().numpy()
    if cmin == cmax:
        cmin=numpy.min(damp)
        cmax=numpy.max(damp)
    fig = plt.figure(facecolor=[0, 32/255, 96/255],figsize=(12,6))
    plt.pcolormesh(damp, vmin=cmin, vmax=cmax,cmap= 'jet')
    ax = plt.gca() 
    ax.invert_yaxis() 
    plt.tick_params(axis='x',colors='w')
    plt.tick_params(axis='y',colors='w')
    ax.spines['right'].set_color('w')
    ax.spines['left'].set_color('w')
    ax.spines['bottom'].set_color('w')
    ax.spines['top'].set_color('w')
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
#     plt.xlabel("step",fontsize=20)
#     plt.ylabel("rate",fontsize=20)
    cb=plt.colorbar(pad=0.03)
    cb.ax.tick_params(colors='w',labelsize=18)
    cb.ax.spines['right'].set_color('w')
    matplotlib.cm.get_cmap('jet')
def extend_wave(s,nt):
    ng = np.size(s,axis=1)
    if np.size(s[:,0])<nt:
        ss=np.zeros((nt,ng))
        for i in range(ng):
            xr=np.zeros((nt))
            xr[0:np.size(s[:,0])]=s[:,i]
            ss[:,i]=xr
    return ss
def pad(p0, nbc, pad_top):
    p=torch.cat((p0[:,0].reshape(-1,1).repeat(1,nbc),p0,p0[:,-1].reshape(-1,1).repeat(1,nbc)),dim=1)
    p=torch.cat((p[0,:].reshape(1,-1).repeat(pad_top,1),p,p[-1,:].reshape(1,-1).repeat(nbc,1)),dim=0)
    return p
def damp_circle(vmin,nzbc,nxbc,nbc,dx,isfs,pad_top):
    nz=nzbc-nbc-pad_top
    nx=nxbc-2*nbc

    a=(nbc-1)*dx
    kappa = 3.0 * vmin * 16.118095650958320 / (2.0 * a)

    # setup 1D BC damping array
    damp1d=kappa*(torch.arange(0,nbc)*dx/a)**2
    damp=torch.zeros((nzbc,nxbc))


# Take care of the 4 boundaries
   # Left and right
    for iz in range(pad_top,nz+pad_top):
        damp[iz,:nbc]=torch.flip(damp1d,[0])
        damp[iz,nx+nbc:nx+2*nbc]=damp1d


    for ix in range(nbc,nx+nbc):
        if (isfs):
            damp[0:pad_top,ix]=0.0
        else:
            damp[:pad_top,ix]=torch.flip(damp1d,[0])
        damp[nzbc-nbc:nzbc,ix]=damp1d

    # Take care of the 4 corners
        # Upper left
    if (~isfs):
        for iz in range(pad_top):
            for ix in range(nbc):
                dist=math.sqrt((ix-nbc-1)**2+(iz-pad_top-1)**2)
                damp[iz,ix]=kappa*(dist/nbc)**2

         # Upper right
        for iz in range(pad_top):
            for ix in range(nx+nbc,nxbc):
                dist=math.sqrt((ix-nx-nbc)**2+(iz-pad_top-1)**2)
                damp[iz,ix]=kappa*(dist/nbc)**2

   
       # Lower left
    for iz in range(nz+pad_top,nzbc):
        for ix in range(nbc):
            dist=math.sqrt((ix-nbc)**2+(iz-nz-pad_top)**2)
            damp[iz,ix]=kappa*(dist/nbc)**2
       # Lower right
    for iz in range(nz+pad_top,nzbc):
        for ix in range(nx+nbc,nxbc):
            dist=math.sqrt((ix-nbc-nx)**2+(iz-nz-pad_top)**2)
            damp[iz,ix]=kappa*(dist/nbc)**2

    return damp
import torch.nn.functional as F
def staggeredfd_py_cuda(
    inputs,
    temp,
    ca,       
    cl,       
    cm,       
    cm1,        
    b,      
    b1,      
    s,
    device):
    nt = int(inputs[0])
    nzbc = int(inputs[1])
    nxbc = int(inputs[2])
    dtx = float(inputs[3])
    ng = int(inputs[4])
    sz = int(inputs[5]);sz = sz - 1;
    sx = int(inputs[6]);sx = sx - 1;
    gz = int(inputs[7]);gz = gz - 1;
    gx = int(inputs[8]);gx = gx - 1;
    dg = int(inputs[9])
    source_type_num = int(inputs[10])
    fd_order_num = int(inputs[11])
    number_elements = nt*ng
    length_geophone = ng*dg
    nt_interval = int(inputs[12])
    nz = int(inputs[13])
    nx = int(inputs[14])
    format_num = int(inputs[15])
    nbc = (nxbc-nx)//2
    num_nt_record = nt//nt_interval
    wavefield_elements = num_nt_record*nx*nz


    #   Input variables from python numpy: temp ca cl cm b s
    # libtorch Initialising input variables: uu, ww, xx, xz, zz
    uu = torch.zeros((nzbc,nxbc), dtype=torch.float32, device=torch.device('cuda', device))
    ww = torch.zeros((nzbc,nxbc), dtype=torch.float32, device=torch.device('cuda', device))
    xx = torch.zeros((nzbc,nxbc), dtype=torch.float32, device=torch.device('cuda', device))
    xz = torch.zeros((nzbc,nxbc), dtype=torch.float32, device=torch.device('cuda', device))
    zz = torch.zeros((nzbc,nxbc), dtype=torch.float32, device=torch.device('cuda', device))
    #  libtorch Initialising input variables: fux, fuz, bwx, bwz
    fux = torch.zeros((nzbc,nxbc), dtype=torch.float32, device=torch.device('cuda', device))
    fuz = torch.zeros((nzbc,nxbc), dtype=torch.float32, device=torch.device('cuda', device))
    bwx = torch.zeros((nzbc,nxbc), dtype=torch.float32, device=torch.device('cuda', device))
    bwz = torch.zeros((nzbc,nxbc), dtype=torch.float32, device=torch.device('cuda', device))
    illum_div =  torch.zeros((nzbc,nxbc), dtype=torch.float32, device=torch.device('cuda', device))
    #  libtorch Initialising output variables: seismo_w, seismo_u       
    seismo_w = torch.zeros((nt,ng), dtype=torch.float32, device=torch.device('cuda', device))
    seismo_u = torch.zeros((nt,ng), dtype=torch.float32, device=torch.device('cuda', device))

    wavefield_gradient_fux = torch.zeros((nz,nx*num_nt_record), dtype=torch.float32, device=torch.device('cuda', device))
    wavefield_gradient_fuz = torch.zeros((nz,nx*num_nt_record), dtype=torch.float32, device=torch.device('cuda', device))
    wavefield_gradient_bwx = torch.zeros((nz,nx*num_nt_record), dtype=torch.float32, device=torch.device('cuda', device))
    wavefield_gradient_bwz = torch.zeros((nz,nx*num_nt_record), dtype=torch.float32, device=torch.device('cuda', device))

    #  libtorch zero_vector for free surface zz
    zero_vector = torch.zeros((1,nxbc), dtype=torch.float32, device=torch.device('cuda', device))
    geophone_vector = torch.zeros((1,nxbc), dtype=torch.float32, device=torch.device('cuda', device))

    if fd_order_num==22 :
        k = nzbc-2; i = nxbc-2; pad_top = 1;
    elif fd_order_num==24 :
        k = nzbc-4; i = nxbc-4; pad_top = 2;
    elif fd_order_num==26 :
        k = nzbc-6; i = nxbc-6; pad_top = 3;
    elif fd_order_num==28 :
        k = nzbc-8; i = nxbc-8; pad_top = 4;
    S41 = 1.1250;S42 = -0.0416666667;S61 = 1.17187;S62 = -6.51042E-2;S63 = 4.68750E-3;S81 = 1.19629;S82 = -7.97526E-2;S83 = 9.57031E-3;S84 = -6.97545E-4;

    for it in range(nt) :
        if  fd_order_num == 22:
            uu = F.pad((temp[1:1+k, 1:1+i]*(uu[1:1+k, 1:1+i]) + b[1:1+k, 1:1+i]*(  \
                    xx[1:1+k, 1+1:1+1+i] - xx[1:1+k, 1:1+i] + xz[1:1+k, 1:1+i] - xz[1-1:1-1+k, 1:1+i])), [1, 1, 1, 1])   
            ww = F.pad((temp[1:1+k, 1:1+i]*(ww[1:1+k, 1:1+i]) + b1[1:1+k, 1:1+i]*(  \
                    xz[1:1+k, 1:1+i] - xz[1:1+k, 1-1:1-1+i] + zz[1+1:1+1+k, 1:1+i] - zz[1:1+k, 1:1+i])), [1, 1, 1, 1])         
        
        if source_type_num == 3:
            uu[sz,sx] = uu[sz,sx] + s[it]
            ww[sz,sx] = ww[sz,sx] + s[it]
        elif source_type_num == 5:
            ww[sz,sx]=s[it]    
            
        if fd_order_num == 22:
            fux = F.pad((uu[1:1+k, 1:1+i] - uu[1:1+k, 1-1:1-1+i]), [1, 1, 1, 1])   
            fuz = F.pad((uu[1+1:1+1+k, 1:1+i] - uu[1:1+k, 1:1+i]), [1, 1, 1, 1])   
            bwx = F.pad((ww[1:1+k, 1+1:1+1+i] - ww[1:1+k, 1:1+i]), [1, 1, 1, 1])   
            bwz = F.pad((ww[1:1+k, 1:1+i] - ww[1-1:1-1+k, 1:1+i]), [1, 1, 1, 1])   
            
        xx=temp * (xx) + (ca * (fux) + cl * (bwz))*dtx
        zz=temp * (zz) + (ca * (bwz) + cl * (fux))*dtx
        xz=temp * (xz) + (cm1 * (fuz + bwx))*dtx

        zz[pad_top,:]=0.0

        seismo_w[it, :] = (1+0*torch.randn((1,ng),device=torch.device('cuda', device))) * ww[gz,gx:gx+length_geophone-1:dg]

        seismo_u[it, :] = (1+0*torch.randn((1,ng),device=torch.device('cuda', device))) * uu[gz,gx:gx+length_geophone-1:dg]
        if(it%nt_interval==0):
            illum_div = illum_div+(fux.data + bwz.data)**2
            wavefield_gradient_fux[:, nx*it//nt_interval:nx*it//nt_interval + nx]=fux[pad_top+1:pad_top+1+nz, nbc:nbc + nx]
            wavefield_gradient_fuz[:, nx*it//nt_interval:nx*it//nt_interval + nx]=fuz[pad_top+1:pad_top+1+nz, nbc:nbc + nx]
            wavefield_gradient_bwx[:, nx*it//nt_interval:nx*it//nt_interval + nx]=bwx[pad_top+1:pad_top+1+nz, nbc:nbc + nx]
            wavefield_gradient_bwz[:, nx*it//nt_interval:nx*it//nt_interval + nx]=bwz[pad_top+1:pad_top+1+nz, nbc:nbc + nx]  


    return seismo_u, seismo_w, illum_div
import scipy
def smooth2a(matrixIn,Nr,Nc):
    device = matrixIn.device
    matrixIn = matrixIn.cpu().numpy()
    [row,col] = matrixIn.shape
    eL = scipy.sparse.spdiags(numpy.ones((2*Nr,row)),numpy.arange(-Nr,Nr),row,row)
    eR = scipy.sparse.spdiags(numpy.ones((2*Nc,col)),numpy.arange(-Nc,Nc),col,col)
    nrmlize = eL@(numpy.ones_like(matrixIn))@eR
    matrixOut = eL@matrixIn@eR
    matrixOut = matrixOut/nrmlize
    matrixOut = torch.from_numpy(matrixOut)
    matrixOut = matrixOut.to(device)
    return matrixOut
def ricker(nt,fr,dt):
    nw=2.2/fr/dt
    nw=2*math.floor(nw/2)+1
    nc=math.floor(nw/2)
    k=torch.arange(1,nw+1,1)
    alpha = (nc-k+1)*fr*dt*3.14159265358
    beta=alpha**2
    w0 = (1-beta**2)*torch.exp(-beta)
    s1 = torch.zeros((nt,1))
    s1[0:len(w0),0] = w0
    return s1

def callParameter(vp0,vs0,dtx,nbc,pad_top):
    
    den = pad(torch.ones_like(vp0),nbc,pad_top)
    vp1 = pad(vp0,nbc,pad_top)
    vs1 = pad(vs0,nbc,pad_top)
    ca=torch.mul(vp1**2,den)
    cm=torch.mul((vs1**2),den)
    cl=ca - 2 * cm
    den1=den
    ca[pad_top,:]=2*cm[pad_top,:]
    cl[pad_top,:]=0.0
    cm1=cm
    b=dtx*torch.reciprocal(den)
    b[pad_top,:]=2*b[pad_top,:]
    b1=b
    return den, ca, cm, cl, cm1, b, b1
def GW_mute(seismo_w,T1,gx,sx,T0,vg,dt,ztr):
# zoffset_mask = mute_near_offset(seismo_w,gx-sx,ztr)
    device = seismo_w.device
    tau_o = T0 + torch.ceil(torch.abs(gx-sx)/vg/dt)
    [nt,ng] = seismo_w.shape
    tapering = torch.ones((nt,ng), device = device)
    sig = 0.05* T1
    for k in range(ng):
        b2 = tau_o[k]
        for j in range(nt):
            if j> b2:
                tapering[j,k] = torch.exp((-(j-b2)**2)/(2*(sig**2)))
    seismo_w_mute = seismo_w*(tapering) 
    return seismo_w_mute,tapering
def GW_mute1(seismo_w,T1,gx,sx,T0,vg,dt,ztr):
# zoffset_mask = mute_near_offset(seismo_w,gx-sx,ztr)
    device = seismo_w.device
    tau_o = T0 + torch.ceil(torch.abs(gx-sx)/vg/dt)
    [nt,ng] = seismo_w.shape
    tapering = torch.ones((nt,ng), device = device)
    sig = 0.05* T1
    for k in range(ng):
        b2 = tau_o[k]
        for j in range(nt):
            if j> b2:
                tapering[j,k] = torch.exp((-(j-b2)**2)/(2*(sig**2)))
    seismo_w_mute = seismo_w*(1-tapering) 
    return seismo_w_mute,(1-tapering) 
# def mute_near_offset(input1,h,lamba):
# output=torch.ones_like(input1)
# dh=h[2]-h[1];
# idh=torch.abs(h/dh);
# [~,ix2]=find(idh==lamba)
# if ~isempty(ix2)
#     for ix=1:length(h)
#         if (idh[ix]<=lamba)
#            output[:,ix]=0

# return output
def bp_filter(d,dt,f1,f2,f3,f4):
    device = d.device
    [nt,nx] = d.shape
    k = torch.ceil(torch.log2(torch.tensor([nt])))
    nf = int(4*(2**k))

    i1 = int(math.floor(nf*f1*dt)+1)
    i2 = int(math.floor(nf*f2*dt)+1)
    i3 = int(math.floor(nf*f3*dt)+1)
    i4 = int(math.floor(nf*f4*dt)+1)

    up =  torch.arange(1,i2-i1+1, device = device).reshape(1,-1)/(i2-i1)
    down = torch.arange(i4-i3,0,-1, device = device).reshape(1,-1)/(i4-i3)
    aux = torch.cat([torch.zeros((1,i1), device = device), up, torch.ones((1,i3-i2), device = device),  \
                     down, torch.zeros((1,nf//2+1-i4), device = device)],1) 
    aux2 = torch.flip(aux[:,0:nf//2-1],[1])

    c = 0
    F = torch.transpose(torch.cat([aux,aux2],1), 0, 1)
    Phase = (math.pi/180.)*torch.cat([torch.tensor([0], device = device).reshape(1,-1),-c*torch.ones((1,nf//2-1),device = device),  \
                           torch.tensor([0], device = device).reshape(1,-1),c*torch.ones((1,nf//2-1), device = device)],1)
    Transfer = F*torch.exp(-1j*torch.transpose(Phase, 0, 1))
    Transfer = Transfer.reshape(nf)

    D = torch.fft.fft(d,nf,0)
    Do = torch.zeros_like(D,device = device)
    for k in range(nx):
        Do[:,k] = Transfer*D[:,k]


    o = torch.fft.ifft(Do,nf,0)

    o = torch.real(o[0:nt,:])
    return o