from math import exp, log, e
import numpy as np
from numpy import random
from util_tool import get_node_information,read_dict,save_dict
from walk_core_model import core_model
from pandas import DataFrame
from evaluate import evaluate_tools
from matplotlib import pyplot as plt
import os
import time

class struct2vec(core_model):

    def __init__(self, Graph, per_vertex, walk_length, window_size, dimension_size, work, k , path='struct_all_node_k_degree.txt'):
        super().__init__(Graph, per_vertex, walk_length, window_size, dimension_size, work)
        self.idx2node, self.node2idx = get_node_information(self.all_nodes)
        self.k = k
        if not os.path.exists(path):
            print('init node degree list')
            print('loading...')
            start_time=time.time()

            self.all_node_k_degree = self.generator_batch_node_k_degree(self.all_nodes)
            save_dict(self.all_node_k_degree,path)
            print('init degree list latering:{}'.format(time.time()-start_time))
        else:
            self.all_node_k_degree=read_dict(path)
        self.generator_all_node_degree()

    def generator_node_k_degree(self, now_vertex, opt1=True):
        pre_node_rank_list = []
        now_node_rank_list = [self.node2idx[now_vertex]]
        level = 0
        all_degree_list = {}

        while level < self.k:
            path = []
            degree_list = []
            for now_node_rank in now_node_rank_list:
                neighbors_list = list(self.G.neighbors(self.idx2node[now_node_rank]))

                neighbors_list = [self.node2idx[neighbors] for neighbors in neighbors_list]

                for pre_node in pre_node_rank_list:
                    if pre_node in neighbors_list:
                        neighbors_list.remove(pre_node)

                degree_list.append(len(neighbors_list))

                path += neighbors_list
            all_degree_list[level] = degree_list

            pre_node_rank_list = now_node_rank_list
            now_node_rank_list = path
            now_node_rank_list = np.unique(now_node_rank_list)
            level += 1

        if opt1:

            for key in all_degree_list:
                value_list = all_degree_list[key]
                value_list, sequence_list = np.unique(value_list, return_counts=True)
                all_degree_list[key] = [(val, sequence) for val, sequence in zip(value_list, sequence_list)]
                if all_degree_list[key]==[]:
                    all_degree_list[key]=[(0,0)]

        return all_degree_list

    def generator_batch_node_k_degree(self, batch_nodes):
        all_k_node_degree = {}

        num=len(batch_nodes)
        count=1
        for node in batch_nodes:
            print('now:{},rest{}'.format(count,num-count))
            degree_list = self.generator_node_k_degree(node,self.k)
            all_k_node_degree[self.node2idx[node]] = degree_list
            count+=1
        self.all_k_degree_list = all_k_node_degree

        return all_k_node_degree

    def binary_find(self, node, node_list):

        start = 0
        end = len(node_list)
        mid = (start + end) // 2

        node_path = []

        while node != node_list[mid]:
            node_path.append(mid)

            if node < node_list[mid]:
                end = mid
                mid = (start + end) // 2
            else:
                start = mid
                mid = (start + end) // 2
            if start == end - 1:
                break

        return node_path

    def generator_all_node_degree(self):

        all_node_degree_rank = []
        all_node_degree_value = []

        for node in self.all_nodes:
            node_degree = len(list(self.G.neighbors(node)))
            all_node_degree_rank.append(self.node2idx[node])
            all_node_degree_value.append(node_degree)

        degree_information = DataFrame()
        degree_information['node_rank'] = all_node_degree_rank
        degree_information['node_degree'] = all_node_degree_value
        degree_information.sort_values(by=['node_degree'], inplace=True)

        self.degree_information = degree_information

    def pair_elem_distance(self, elem_u, elem_v, opt1=True):
        if not opt1:
            if min(elem_u, elem_v) == 0:
                return 0
            return max(elem_u, elem_v) / min(elem_u, elem_v) - 1
        else:
            u_degree, u_sequence = elem_u
            v_degree, v_sequence = elem_v

            if min(u_degree, v_degree) == 0:
                return 0

            return (max(u_degree, v_degree) / min(u_degree, v_degree) - 1) * max(u_sequence, v_sequence)

    def dtw(self, quences_A, quences_B):
        if quences_A == []:
            quences_A = [(0, 0)]
        if quences_B == []:
            quences_B = [(0, 0)]

        M = len(quences_A)
        N = len(quences_B)

        init_x, init_y = 0, 0
        now_x, now_y = init_x, init_y
        path = [(now_x, now_y)]
        all_cost = self.pair_elem_distance(quences_A[now_x], quences_B[now_y])

        while True:
            if now_x == M - 1:
                if now_y == N - 1:
                    break

            next_x_list = list(range(now_x, min(now_x + 2, M)))
            next_y_list = list(range(now_y, min(now_y + 2, N)))

            min_cost = float('Inf')
            min_point_x = 0
            min_point_y = 0

            for x in next_x_list:
                for y in next_y_list:
                    if not (x, y) in path:
                        cost = self.pair_elem_distance(quences_A[x], quences_B[y])
                        if x == y:
                            cost *= 2
                        if cost < min_cost:
                            min_cost = cost
                            min_point_x = x
                            min_point_y = y

            all_cost += min_cost
            path.append((min_point_x, min_point_y))
            now_x, now_y = min_point_x, min_point_y

        return all_cost

    def get_rank_list_data(self, rank_list, data_dict):
        aim_data = {}
        for rank in rank_list:
            aim_data[rank] = data_dict[rank]
        return aim_data

    def constructing_context_graph(self, now_vertex,opt2=True):
        node_rank = self.node2idx[now_vertex]
        normal_information = self.degree_information[self.degree_information['node_rank'] != node_rank]

        if opt2:
            node_degree = len(list(self.G.neighbors(now_vertex)))
            node_list = normal_information['node_degree'].tolist()
            path_rank = self.binary_find(node_degree, node_list)
            normal_node_rank_list = self.degree_information['node_rank'][path_rank].tolist()
        else:
            normal_node_rank_list = normal_information['node_rank']

        batch_normal_k_node_degree_list = self.get_rank_list_data(normal_node_rank_list, self.all_node_k_degree)
        normal_k_node_degree_list = self.get_rank_list_data([node_rank], self.all_node_k_degree)[node_rank]

        all_node_weights = {}

        for node_key in list(batch_normal_k_node_degree_list.keys()):
            node_weight = {}
            node_v_degree_list = batch_normal_k_node_degree_list[node_key]
            for level in range(self.k):
                distance = self.dtw(normal_k_node_degree_list[level], node_v_degree_list[level])
                weight = exp(-1 * distance)
                node_weight[level] = weight
            all_node_weights[node_key] = node_weight

        return all_node_weights

    def compute_move_layers_probability(self, all_node_weights):
        layer_avg_weights = {}
        for level in range(self.k):
            avg_weights = 0
            num_count = 0
            for node in all_node_weights.keys():
                avg_weights += all_node_weights[node][level]
                num_count += 1

            if num_count != 0:
                layer_avg_weights[level] = avg_weights / num_count
            else:
                layer_avg_weights[level] = 0

        up_layer_weight = {}
        down_layer_weight = {}

        for level in range(self.k):
            num_node = 0
            for node in all_node_weights.keys():
                weight = all_node_weights[node][level]
                if weight > layer_avg_weights[level]:
                    num_node += 1
            up_layer_weight[level] = log(num_node + e)
            down_layer_weight[level] = 1

        return up_layer_weight, down_layer_weight, layer_avg_weights

    def generator_context_for_nodes(self, up_layer_weight, down_layer_weight, all_node_weights, layer_avg_weights):

        probability_layer_nodes = {}
        node_rank = list(all_node_weights.keys())

        for level in range(self.k):
            level_node_probability = []
            for node in all_node_weights.keys():
                if layer_avg_weights[level] == 0:
                    probability = 0
                else:
                    probability = all_node_weights[node][level] / layer_avg_weights[level]
                level_node_probability.append(probability)
            prab, alias = self.optimize_fun.generate_alias_table(level_node_probability)
            probability_layer_nodes[level] = [prab, alias]

        probability_up_move = {}
        probability_down_move = {}

        for level in range(self.k):
            up_ = up_layer_weight[level]
            down_ = down_layer_weight[level]

            probability_up = up_ / (up_ + down_)
            probability_down = 1 - probability_up

            probability_up_move[level] = probability_up
            probability_down_move[level] = probability_down

        return probability_layer_nodes, node_rank, probability_up_move, probability_down_move

    def prepare_bias_random_walk(self, now_vertex):
        all_node_weights = self.constructing_context_graph(now_vertex)
        up_layer_weight, down_layer_weight, layer_avg_weights = self.compute_move_layers_probability(
            all_node_weights)
        probability_layer_nodes, node_rank, probability_up_move, probabilty_down_move = self.generator_context_for_nodes(
            up_layer_weight=up_layer_weight,
            all_node_weights=all_node_weights,
            down_layer_weight=down_layer_weight,
            layer_avg_weights=layer_avg_weights
        )

        return probability_layer_nodes, node_rank, probability_up_move

    def bias_random_walk(self, now_vertex):
        probability_layer_nodes, node_rank_list, probability_up_move = self.prepare_bias_random_walk(now_vertex)
        level = 0

        path = [now_vertex]

        while self.sentence_len != len(path):
            if len(path) == 1:
                prab, alias = probability_layer_nodes[level]

            print(prab)
            print('******************')
            rank_index = self.optimize_fun.alias_sample(prab, alias)
            node_rank = node_rank_list[rank_index]
            path.append(self.idx2node[node_rank])

            while True:

                r = random.random()
                up_ = probability_up_move[level]

                if r > up_:
                    if level != self.k - 1:
                        level += 1
                else:
                    if level != 0:
                        level -= 1

                prab, alias = probability_layer_nodes[level]
                if prab!=[]:
                    break

            print(prab)
            print(path)
            print('----------------------')

        return path

    def struct_walk(self):
        sentence_list = []
        all_node = self.all_nodes

        for step in range(self.walk_epoch):
            random.shuffle(all_node)
            for node in all_node:
                sentence = self.bias_random_walk(node)
                sentence_list.append(sentence)

        return sentence_list

    def train(self):
        sentence_list = self.struct_walk()
        embeddings = self.embdding_train(sentence_list)
        evaluate_tools(embeddings).plot_embeddings()
        plt.show()

if __name__ == '__main__':
    path = '../wiki/Wiki_edgelist.txt'
    from util_tool import read_graph
    G = read_graph(edgelist_path=path)
    model = struct2vec(
        Graph=G,
        walk_length=10,
        k=8,
        window_size=5,
        dimension_size=128,
        per_vertex=10,
        work=4
    )
    model.train()




















