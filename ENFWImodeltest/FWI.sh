#!/bin/bash
# 
# 使用pytorch distribute data parallel 进行二维有限差分弹性波FWI。正演过程由数值计算抽象为网络，网络权重为Vp Vs速度结构。
# 此bash程序调度分配gpu，并启动反演程序。
# 调度例子：假设一共有20炮数据反演，8个gpu可被使用，因此一次命令最多启动8个进程。2*8+4=20 
# 因此运行三遍命令启动程序，前两次启动8个进程，最后一次启动4个进程。也可以2*7+6=20.
# 输入参数： w 总进程数量； 
#          n gpu数量； 
#          m 一个gpu最多进程个数（取决于模型大小和GPU显存）
# Two dimensional finite difference elastic wave FWI using pytorch distribute data parallel. 
# the forward process is changed from numerical computation to a network structure. and Vp Vs velocity structure is network weights.
# This bash program allocates gpu and procs and starts the inversion process.
# allocates example: suppose we have a total of 20 shots of data . and 8 gpu can be used, so one command starts at most 8 processes. 2*8+4=20 
# So run the command three times to start the program, the first two times to start 8 processes, the last time to start 4 processes. Also 2*7+6=20 work
# .
# Input parameters: w Total number of processes; 
#                   n Number of gpu; 
#                   m maximum number of processes on a gpu (depends on model size and GPU memory)

while getopts 'w:n:m:' OPT; do
    case $OPT in
        w) world_size=$OPTARG;;
        n) numgpu=$OPTARG;;
        m) max_forward=$OPTARG;;
    esac
done

echo "pytorch DDP FWI"
python=/home/zhangchang/miniconda3/bin/python
program=/home/zhangchang/python/ENFWImodeltest/ENFWImodeltest.py
all_forwrad=$[world_size/numgpu]
ranklast=$((world_size%numgpu))
if [ $ranklast -ne 0 ];then
     all_forwrad=$((all_forwrad+1))
fi

if [ $all_forwrad -ge $max_forward ];then
     forward_num=${max_forward}
else
     forward_num=${all_forwrad}
fi

if [ $ranklast -eq 0 ];then
     last_proc=${numgpu}
else
     last_proc=${ranklast}
fi
sub_forward=$((all_forwrad-forward_num+1))

echo "world_size: ${world_size}" # 开启总进程数 number of total porc
echo "numgpu: ${numgpu}" # gpu数量
echo "all_forwrad: ${all_forwrad}" 
echo "ranklast: ${ranklast}" # 
echo "sub_forward: ${sub_forward}"
echo "max_forward: ${max_forward}" # 一个gpu最大进程数 max porc in one gpu
echo "forward_num: ${forward_num}" # 启动程序的次数 Number of the command was started
echo "last_proc: ${last_proc}"
addr="--world_size=${world_size} --forward_num=${forward_num} --master_addr="127.0.0.1" --master_port=32495"
command=""
for ((i=0;i<$((forward_num-1));i++))
do
echo "$i"
# command="${command}"123""  --sub_forward=${sub_forward}
command="${command} ${python} ${program} --node_rank=${i} --num_proc=${numgpu} --sub_forward=${sub_forward} ${addr} & "
done
echo "$i"
command="${command} ${python} ${program} --node_rank=${i} --num_proc=${last_proc} --sub_forward=${sub_forward} ${addr}"
echo "${command}"
eval "${command}"