import numpy as np
import networkx as nx
import random
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from matplotlib import colors as mcolors
from scipy.spatial import distance
from collections import defaultdict
import copy
class springgreen():
    def __init__(self, nx_G, is_directed , p_jump ,similarity = None, similarity_type =None , similarity_th = None):
        self.G = nx_G
        self.is_dirceted = is_directed
        self.similarity = similarity
        self.similarity_type = similarity_type
        self.p_jump = p_jump
        self.similarity_th = similarity_th
        
    
    def preprocess_weight(self):
        # 
        G = self.G
        neighbor_probs = {}
        for node in G.nodes():
            probs = [G[node][nbr]['weight'] for nbr in sorted(G.neighbors(node))]
            norm_const = sum(probs)
            normalized_probs =  [float(u_prob)/norm_const for u_prob in probs]
            neighbor_probs[node] = normalized_probs
        self.neighbor_probs = neighbor_probs
    
    
    def preprocess_similarity(self, top_k =0):
        similarity_type = self.similarity_type
        if similarity_type == 'random':
            self.random_similarity()
        elif similarity_type == 'degree':
            self.degree_similarity()
        elif similarity_type == 'identity':
            self.identity_similarity()
        elif similarity_type == 'neighbor':
            self.neighbor_similarity()
        elif similarity_type == 'binary':
            self.binary_similarity()
        elif similarity_type == 'dict':
            self.dict_similarity()
        elif similarity_type == None:
            similarity = self.similarity
            G = self.G
            nodes = list(G.nodes())
            size = G.number_of_nodes()
            jump_destination = dict()
            try:
                if similarity.any():
        ####### This part need to be complete
                    for node in nodes:
                        jump_destination[node] = set()
                        ct = 1
                        for i in reversed(np.argsort(similarity[node-1])):
                            if ct > max_s:
                                break
                            else:
                                if similarity[node-1][i]>0:
                                    jump_destination[node].add(i+1)
                                    ct +=1
                    self.jump_destination = jump_destination
            except:
                print( 'you need to specifies either similarity or similarity_type')
        if top_k != 0:
            print('top_k similarity is choosen')
            self.top_k_similar( top_k)
    
    def random_similarity(self):

        #### I don't satisfied with the code here( ToDo: change the code)
        G = self.G
        jump_dist = {}
        size = G.number_of_nodes()
        nodes = sorted(G.nodes())
        for node in nodes:
            v = np.random.randn(size,1)
            v = v*(v >0)
            v = v*1/sum(v)
            jump_dist[node]= {}
            jump_dist[node]['nodes'] = list()
            jump_dist[node]['probs'] = list()
            for i in range(size):
                if v[i] >0 :
                    jump_dist[node]['nodes'].append(nodes[i])
                    jump_dist[node]['probs'].append(float(v[i]))
        self.jump_dist = jump_dist
        # this part should deleted
        #jump_destination = dict()
        #for node in nodes:
        #    jump_destination[node] = set()
        #    jump_destination[node].add(node)
        #    l = random.randint(0,size/2)
        #    jump_destination[node] = jump_destination[node].union(random.sample(nodes,l))
        #self.jump_destination = jump_destination
        
    def degree_similarity(self):
        G = self.G
        jump_dist = {}
        size = G.number_of_nodes()
        nodes = sorted(G.nodes())
        partition = defaultdict(set)
        for node in nodes:
            degree = len(G.neighbors(node))      
            partition[degree].add(node)
        for value in partition.values():
            l = len(value)
            probs = list(np.ones(l)*1.0/l)
            for node in value:
                jump_dist[node]= {}
                jump_dist[node]['nodes'] = list(value)
                jump_dist[node]['probs'] = probs
        self.jump_dist = jump_dist        
        ### delete this part
        #for value in partition.values():
        #    for node in value:
        #        jump_destination[node] = set(value)
        #self.jump_destination = jump_destination
        
    def identity_similarity(self):
        G = self.G
        jump_destination = dict()
        nodes = list(G.nodes())
        for node in nodes:
            jump_destination[node] = set()
            jump_destination[node].add(node)
        self.jump_destination = jump_destination
    
    def neighbor_similarity(self):
        G = self.G
        jump_destination = dict()
        nodes = list(G.nodes())
        for node in nodes:
            jump_destination[node] = set()
            jump_destination[node].add(node)
            jump_destination[node] = jump_destination[node].union(set(G.neighbors(node)))
        self.jump_destination = jump_destination
        
    def binary_similarity(self):
        try:
            G = self.G
            S = self.similarity
            X = S.dot(S.T)
            jump_dist = {}
            nodes = sorted(G.nodes())
            for index in range(len(nodes)):
                node = nodes[index]
                jump_dist[node]= {}
                jump_dist[node]['nodes'] = []
                jump_dist[node]['probs'] = []
                s1 = set(S[index,:].nonzero()[1])
                for i in X[index,:].nonzero()[1]:
                    s2 = set(S[i,:].nonzero()[1])
                    prob = float(len(s1&s2))/len(s1|s2)
                    if prob > 0:
                        jump_dist[node]['nodes'].append(nodes[i])    
                        jump_dist[node]['probs'].append(prob)
                jump_dist[node]['probs'] = np.array(jump_dist[node]['probs'])/sum(jump_dist[node]['probs'])
            self.jump_dist = jump_dist
        except:
            print('Error in the process of binary similarity , make sure the similarity matrix is given')

        
   
    def dict_similarity(self):
        print('>> Calculating jump distribution base on node similarity...')
        try:
            G = self.G
            S = self.similarity
            nodes = sorted(G.nodes())
            n = len(nodes)
            jump_dist = dict()
            for index1 in range(n):
                node1 = nodes[index1]
                jump_dist[node1]= {}
                jump_dist[node1]['nodes'] = []
                jump_dist[node1]['probs'] = []
                for index2 in range(n):
                    if index2 != index1:
                        node2 = nodes[index2]
                        prob = float(sum(S[node1]&S[node2]))/sum(S[node1]|S[node2])
                        if prob > 0:
                            jump_dist[node1]['nodes'].append(node2)
                            jump_dist[node1]['probs'].append(prob)
                jump_dist[node1]['probs'] = np.array(jump_dist[node1]['probs'])/sum(jump_dist[node1]['probs'])
            self.jump_dist = jump_dist
        except:
            print('Error in the process of dict similarity , make sure the similarity matrix is given')


    def top_k_similar(self, t):
        jump_dist = self.jump_dist
        for node in jump_dist.keys():
            L = jump_dist[node]['probs']
            index = sorted(range(len(L)), key=lambda k: L[k])[-t:]
            s = sum(np.array(jump_dist[node]['probs'])[index])
            jump_dist[node]['probs'] = np.array(jump_dist[node]['probs'])[index]/s
            jump_dist[node]['nodes'] = np.array(jump_dist[node]['nodes'])[index]
        self.jump_dist = jump_dist

    def draw_graph(self, size = 5, dpi =150, node_col= False ):
        G= self.G
        plt.figure(num=None, figsize=(size, size), dpi=dpi)
        labels = nx.get_edge_attributes(G,'weight')
        pos= nx.spring_layout(G)
        if node_col == False:
            nx.draw_networkx(G,pos)
            nx.draw_networkx_edges(G,pos)
            nx.draw_networkx_edge_labels(G,pos = pos,edge_labels= labels,edge_color='b')
        else:
            nodes = self.representation['nodes']
            clusters = self.clusters
            cluster_id = np.unique(clusters)
            color_id = ['red','green','blue','darkslategrey','darkorange', 'darkgrey','springgreen']
            if len(color_id)<len(cluster_id):
                colors = set(dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)) - set(color_id)
                color_id.extend(random.sample(colors,len(cluster_id)-len(color_id)))
                print(len(color_id)),
                print(color_id)
                print(len(cluster_id))
            #color_id = random.sample(colors,len(cluster_id))
            #print (color_id)
            for c_c in zip(cluster_id,color_id):
                cluster_nodes_index = [i for i,val in enumerate(clusters) if val==c_c[0]]
                cluster_nodes = [nodes[i] for i in cluster_nodes_index]
                if(len(cluster_nodes)==0):
                    print('cluster is empty')
                nx.draw_networkx_nodes(G,pos,
                                   nodelist=cluster_nodes,
                                   node_color=c_c[1])
            nx.draw_networkx_edges(G,pos)
    
    def node_sequence_generator(self, jump_num, jump_len, jump_type):
        G = self.G
        node_sequence = []
        nodes = list(G.nodes())
        print('Random Jump Iteration with probabilty %s : '%self.p_jump)
        for  jump_iter in range(jump_num):
            print(str(jump_iter+1), '/', str(jump_num))
            random.shuffle(nodes)
            for node in nodes:
                if jump_type == 'fixed':
                    node_sequence.append(self.random_jump(jump_len = jump_len, start_node = node))
                elif jump_type == 'dynamic':
                    node_sequence.append(self.random_dynamic_jump(jump_len = jump_len, start_node = node))
        return node_sequence
    
    def random_dynamic_jump(self,jump_len, start_node, full_coverage = False):
        #print('start')
        G = self.G
        neighbor_probs = self.neighbor_probs
        p_jump = self.p_jump
        #print(start_node)
        #jump_destination = self.jump_destination[start_node]
        jump_dist_nodes = self.jump_dist[start_node]['nodes']
        jump_dist_probs = self.jump_dist[start_node]['probs']
        rand_seq = [start_node]
        if full_coverage:
            n = len(G.nodes())
            covered = set()
            covered.add(start_node)
            while len(covered)< n:
                current_node = rand_seq[-1]
                jump = np.random.choice([0,1], p=[1-p_jump, p_jump])
                if jump:
                    jump_dist_nodes = self.jump_dist[current_node]['nodes']
                    jump_dist_probs = self.jump_dist[current_node]['probs']
                    next = list(np.random.choice(jump_dist_nodes,1, p= jump_dist_probs))
                    covered.add(next)
                    rand_seq.extend(next)
                else:
                    current_neighbor = sorted(G.neighbors(current_node))
                    if len(current_neighbor) >0:
                        next = np.random.choice(current_neighbor, p= neighbor_probs[current_node])
                        covered.add(next)
                        rand_seq.append(next)
                    else:
                        break               
            
        else:    
            while len(rand_seq) < jump_len:
                current_node = rand_seq[-1]
                jump = np.random.choice([0,1], p=[1-p_jump, p_jump])
                if jump:
                    jump_dist_nodes = self.jump_dist[current_node]['nodes']
                    jump_dist_probs = self.jump_dist[current_node]['probs']
                    next = list(np.random.choice(jump_dist_nodes,1, p= jump_dist_probs))
                    rand_seq.extend(next)
                else:
                    current_neighbor = sorted(G.neighbors(current_node))
                    if len(current_neighbor) >0:
                        next = np.random.choice(current_neighbor, p= neighbor_probs[current_node])
                        rand_seq.append(next)
                    else:
                        break
        return rand_seq


    
    def random_jump(self,jump_len, start_node):
        G = self.G
        neighbor_probs = self.neighbor_probs
        p_jump = self.p_jump
        #print(start_node)
        #jump_destination = self.jump_destination[start_node]
        jump_dist_nodes = self.jump_dist[start_node]['nodes']
        jump_dist_probs = self.jump_dist[start_node]['probs']
        rand_seq = [start_node]
        
        while len(rand_seq) < jump_len:
            current_node = rand_seq[-1]
            jump = np.random.choice([0,1], p=[1-p_jump, p_jump])
            if jump:
                next = list(np.random.choice(jump_dist_nodes,1, p= jump_dist_probs))
                rand_seq.extend(next)
            else:
                current_neighbor = sorted(G.neighbors(current_node))
                if len(current_neighbor) >0:
                    
                    next = np.random.choice(current_neighbor, p= neighbor_probs[current_node])
                    rand_seq.append(next)
                else:
                    break
        return rand_seq

    def W2V_rep(self ,jump_num = 10, jump_len = 80, size=120, window= 10, min_count=0,sg=1, workers=8, iter=1,jump_type = 'fixed',given_sequences=None, generate_sequence= True):
        if generate_sequence:
        	sequences = self.node_sequence_generator(jump_num = jump_num, jump_len= jump_len, jump_type = jump_type)
        else:
        	sequences = given_sequences
        sequences = [map(str, seq) for seq in sequences]
        #print(sequences)
        model = Word2Vec(sequences, size= size, window= window, min_count=min_count, sg=sg, workers=workers, iter=iter)
        indexes = model.vocab.keys()
        embd = list()
        for index in indexes:
            embd.append(model[index])
        embd = np.array(embd)
        indexes = [ int(x) for x in indexes]
        representation = dict()
        representation['nodes'] = indexes
        representation['embeding']= embd
        self.representation = representation
    
    def TSNE_draw(self,size = 5, dpi =100, label= True):
        rep = self.representation
        if rep:
            tsne = TSNE(n_components=2, random_state=0)
            #np.set_printoptions(suppress=True)
            data = tsne.fit_transform(rep['embeding']) 
            X = data[:,0]
            Y = data[:,1]
            labels = rep['nodes']
            fig = plt.figure(figsize=(size, size), dpi=dpi)
            ax = fig.add_subplot(111)
            plt.plot(X,Y, 'ro')
            ct=0
            if label:
                for xy in zip(X,Y):
                    ax.annotate(labels[ct],xy=xy)
                    ct+=1
        else:
            print('no represeantion provided')
            
    def graph_stats(self):
        G = self.G
        print("radius: %d" % nx.radius(G))
        print("diameter: %d" % nx.diameter(G))
        #print("eccentricity: %s" % nx.eccentricity(H))
        print("center: %s" % nx.center(G))
        #print("periphery: %s" % nx.periphery(ER))
        print("density: %s" % nx.density(G))
        partition = defaultdict(set)
        for node in G.nodes():
            degree = len(G.neighbors(node))        
            partition[degree].add(node)
        for key in sorted(partition.keys()):
            print("degree %d:" % key),
            print(partition[key])
    def clustering(self,approach = 'kmeans', n_c =5):
        rep = self.representation
        if approach == 'kmeans':
            kmeans = KMeans(n_clusters=n_c, random_state=0).fit(rep['embeding'])
            self.clusters = kmeans.labels_

    def negative_edge_sampling(self, sample_size):
        G = self.G
        H = nx.complement(G)
        return random.sample(H.edges(),sample_size)


    def positive_edge_sampling(self,sample_size, connected = True):
        #print('hellow')
        G = self.G.copy()
        is_directed = G.is_directed()
        #print('hey')
        if connected:
            print('Connected Sampling')
            E_sample = list()
            while(len(E_sample) < sample_size):
                    e = random.sample(G.edges(),1)[0]
                    w = G[e[0]][e[1]]['weight']
                    G.remove_edge(e[0],e[1])
                    if nx.is_connected(G):
                        E_sample.append(e)
                    else:
                        G.add_edge(e[0],e[1])
                        G[e[0]][e[1]]['weight'] = w
        else:
            print('Not connected Sampling')
            E_sample = random.sample(G.edges(),sample_size)
            G.remove_edges_from(E_sample)
        if not(is_directed):
            G = G.to_undirected()
        return E_sample, G
    def copy(self):
        return shallow_diving(copy.copy(self))