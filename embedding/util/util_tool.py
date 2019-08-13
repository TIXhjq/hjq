# _*_ coding:utf-8 _*_
import os
import networkx as nx
import pandas as pd
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import queue


def get_node_information(all_nodes):
    node2idx = {}
    idx2node = []
    node_size = 0
    for node in all_nodes:
        node2idx[node] = node_size
        idx2node.append(node)
        node_size += 1
    idx2node = idx2node
    node2idx = node2idx
    return idx2node, node2idx

def save_edgelist(edgelist_list,save_path):
    if os.path.exists(save_path):
        os.remove(save_path)

    file=open(save_path,mode='a+')
    for edgelist in edgelist_list:
        file.writelines(edgelist)

def read_graph(edgelist_path='../wiki/Wiki_edgelist.txt'):
    DG=nx.read_edgelist(
        edgelist_path,
        create_using=nx.DiGraph(),
        nodetype=None,
        data=[('weight',int)]
    )


    return DG

def read_node_label(filename, skip_head=False):
    fin = open(filename, 'r')
    X = []
    Y = []
    while 1:
        if skip_head:
            fin.readline()
        l = fin.readline()
        if l == '':
            break
        vec = l.strip().split(' ')
        X.append(vec[0])
        Y.append(vec[1:])
    fin.close()
    return X, Y

def generator_k_degree(G,k, now_vertex,opt1=True):
    pre_node_list = []
    now_node_list = [now_vertex]
    level = 0
    degree_list = []

    while level < k:

        next_node_list=[]
        for now_node in now_node_list:
            normal_node_list=[]

            for node in list(G.neighbors(now_node)):
                if not node in pre_node_list:
                    normal_node_list.append(node)

            pre_node_list+=normal_node_list
            next_node_list+= normal_node_list

        now_node_list=next_node_list
        degree_list.append(len(now_node_list))
        level += 1


    if opt1:
        value_list,sequence_list=np.unique(degree_list,return_counts=True)
        degree_list={val:seq for val,seq in zip(value_list,sequence_list)}

    return degree_list

def save_dict(data,path='struct_all_node_k_degree.txt'):
    f=open(path,'w')
    f.write(str(data))
    f.close()

def read_dict(path='struct_all_node_k_degree.txt'):
    f=open(path,'r')
    data=eval(f.read())
    f.close()
    return data

def read_label(label_path):
    data=pd.read_csv(label_path,header=None,sep=' ')
    nodes=data[0].tolist()
    label=data[1].tolist()

    return nodes,label


if __name__=='__main__':
    edgelist_list=['1 2\n','1 3\n','1 4\n','2 5\n','3 5\n','4 8\n','5 8\n']
    save_path='test.txt'
    save_edgelist(edgelist_list,save_path)
    DG=read_graph(save_path)
    print(generator_k_degree(DG,3,now_vertex='1'))
    # read_label(save_path)

