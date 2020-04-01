<!--
 * @Author: your name
 * @Date: 2020-04-01 18:31:00
 * @LastEditTime: 2020-04-01 20:45:26
 * @LastEditors: Please set LastEditors
 * @Description: In User Settings Edit
 * @FilePath: \PyTorch-Distributed-Training\README.md
 -->
# PyTorch-Distributed-Training
Example of PyTorch DistributedDataParallel

## Single machine multi gpu
'''
python -m torch.distributed.launch --nproc_per_node=ngpus --master_port=29500 main.py ...
'''

## Multi machine multi gpu
suppose we have two machines and one machine have 4 gpus


In multi machine multi gpu situation, you have to choose a machine to be master node.

we named the machines A and B, and set A to be master node

script run at A

'''
python -m torch.distributed.launch --nproc_per_node=4 --nnode=2 --node_rank=0 --master_addr=A_ip_address master_port=29500 main.py ...
'''

script run at B

'''
python -m torch.distributed.launch --nproc_per_node=4 --nnode=2 --node_rank=1 --master_addr=A_ip_address master_port=29500 main.py ...
'''