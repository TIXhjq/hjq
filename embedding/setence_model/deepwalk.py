# _*_ coding:utf-8 _*_
from numpy import random
from walk_core_model import core_model
from evaluate import evaluate_tools

class DeepWalk(core_model):

    def __init__(self,Graph,per_vertex,walk_length,window_size,dimension_size,work):
        super().__init__(Graph,per_vertex,walk_length,window_size,dimension_size,work)

    def deepwalk(self):
        sentence_list=[]

        for num in range(self.walk_epoch):
            random.shuffle(self.all_nodes)
            for vertex in self.all_nodes:
                sentence_list.append(self.random_walk(start_vertex=vertex))

        return sentence_list

    def transform(self):
        sentence_list=self.deepwalk()
        embeddings=self.embdding_train(sentence_list)
        return embeddings


if __name__=='__main__':
    Graph = read_graph('../wiki/Wiki_edgelist.txt')

    deepwalk=DeepWalk(
        Graph=Graph,
        per_vertex=80,
        walk_length=10,
        window_size=5,
        dimension_size=128,
        work=4
    )

    embeddings=deepwalk.transform()
    eval = evaluate_tools(embeddings=embeddings)
    eval.plot_embeddings()

