import numpy as np
import scipy.sparse as sp


def get_adj(train_data, n_node, path):
    try:
        norm_adj_mat = sp.load_npz('datasets/' +path + '/norm_adj_mat.npz')
        print('already load adj matrix', norm_adj_mat.shape)
    except Exception:
        norm_adj_mat = create_adj_mat(train_data, n_node)
        sp.save_npz('datasets/' +path + '/norm_adj_mat.npz', norm_adj_mat)
    return norm_adj_mat


def create_adj_mat(train_data, n_node):
    adj = sp.dok_matrix((n_node,n_node), dtype=np.float32)
    for seq in train_data:
        for i in range(len(seq) -1 ):
            if adj[seq[i]-1, seq[i+1]-1] ==0:
                adj[seq[i]-1, seq[i+1]-1] = 1
                adj[seq[i+1]-1, seq[i]-1] = 1
            else:
                adj[seq[i]-1, seq[i+1]-1] += 1
                adj[seq[i+1]-1, seq[i]-1] += 1

    adj = adj.todok()

    def normalized_adj_mat(adj):
        rowsum = np.array(adj.sum(1))
        d_inv = np.power(rowsum, -1).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        norm_adj = d_mat_inv.dot(adj)
        return norm_adj.tocoo()

    norm_adj = normalized_adj_mat(adj + sp.eye(adj.shape[0]))
    #mean_adj = normalized_adj_mat(adj)

    return norm_adj.tocsr()


# def get_adj(train_data, n_node, path):
#     try:
#         #adj_mat = sp.load_npz('datasets/' + path + '/adj_mat.npz')
#         adj_mat_in = sp.load_npz('datasets/' + path + '/adj_mat_in.npz')
#         adj_mat_out = sp.load_npz('datasets/' +path + '/adj_mat_out.npz')
#         print('already load adj matrix', adj_mat_out.shape)
#     except Exception:
#         adj_mat_out, adj_mat_in = create_adj_mat(train_data, n_node)
#         #sp.save_npz('datasets/' + path + '/adj_mat.npz', adj_mat)
#         sp.save_npz('datasets/' + path + '/mean_adj_mat.npz', adj_mat_in)
#         sp.save_npz('datasets/' +path + '/norm_adj_mat.npz', adj_mat_out)
#     return adj_mat_out, adj_mat_in


# def create_adj_mat(train_data, n_node):
#     adj = sp.dok_matrix((n_node,n_node), dtype=np.float32)
#     for seq in train_data:
#         for i in range(len(seq) -1 ):
#             if adj[seq[i]-1, seq[i+1]-1] ==0:
#                 adj[seq[i]-1, seq[i+1]-1] = 1
#                 #adj[seq[i+1]-1, seq[i]-1] = 1
#             else:
#                 adj[seq[i]-1, seq[i+1]-1] += 1
#                 #adj[seq[i+1]-1, seq[i]-1] += 1

#     adj = adj.todok()
#     adj_in = adj.T

#     def normalized_adj_mat(adj):
#         rowsum = np.array(adj.sum(1))
#         d_inv = np.power(rowsum, -1).flatten()
#         d_inv[np.isinf(d_inv)] = 0.
#         d_mat_inv = sp.diags(d_inv)
#         norm_adj = d_mat_inv.dot(adj)
#         return norm_adj.tocoo()

#     adj_mat_out = normalized_adj_mat(adj + sp.eye(adj.shape[0]))
#     adj_mat_in = normalized_adj_mat(adj_in + sp.eye(adj_in.shape[0]))

#     return adj_mat_out.tocsr(), adj_mat_in.tocsr()


def data_masks(all_usr_pois, item_tail, truncate=True):

    us_lens = [len(upois) for upois in all_usr_pois]
    len_max = max(us_lens)
    us_pois = [upois + item_tail * (len_max - le) for upois, le in zip(all_usr_pois, us_lens)]
    us_msks = [[1] * le + [0] * (len_max - le) for le in us_lens]
    

    if truncate:
        maxlen=19
        new_pois=[]
        new_mask=[]
        for x,y  in zip(us_pois, us_msks):
            if len(x) < maxlen:
                new_pois.append(x)
                new_mask.append(y)
            else:
                new_pois.append(x[:maxlen])
                new_mask.append(y[:maxlen])

        us_pois = new_pois
        us_msks = new_mask
        #del new_pois, new_mask



        # new_train_set_x = []
        # new_train_set_y = []
        # for x, y in zip(train_set[0], train_set[1]):
        #     if len(x) < maxlen:
        #         new_train_set_x.append(x)
        #         new_train_set_y.append(y)
        #     else:
        #         new_train_set_x.append(x[:maxlen])
        #         new_train_set_y.append(y)

        # train_set = (new_train_set_x, new_train_set_y)
        # del new_train_set_x, new_train_set_y
    
    return us_pois, us_msks, len_max
    #return new_pois, new_mask, len_max


def split_validation(train_set, valid_portion):
    #train_set = self.traindata
    



    train_set_x, train_set_y = train_set
    n_samples = len(train_set_x)
    sidx = np.arange(n_samples, dtype='int32')
    np.random.shuffle(sidx)
    n_train = int(np.round(n_samples * (1. - valid_portion)))
    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]

    return (train_set_x, train_set_y), (valid_set_x, valid_set_y)


class Data():
    def __init__(self, data, shuffle=False, truncate=False):

        self.truncate=truncate
        inputs = data[0]
        inputs, mask, len_max = data_masks(inputs, [0], truncate=self.truncate)
        self.inputs = np.asarray(inputs)
        self.mask = np.asarray(mask)
        self.len_max = len_max
        self.targets = np.asarray(data[1])
        self.length = len(inputs)
        self.shuffle = shuffle


    def generate_batch(self, batch_size):
        if self.shuffle:
            shuffled_arg = np.arange(self.length)
            np.random.shuffle(shuffled_arg)
            self.inputs = self.inputs[shuffled_arg]
            self.mask = self.mask[shuffled_arg]
            self.targets = self.targets[shuffled_arg]
            #self.adj = self.adj[shuffled_arg]
        n_batch = int(self.length / batch_size)
        if self.length % batch_size != 0:
            n_batch += 1
        slices = np.split(np.arange(n_batch * batch_size), n_batch)
        slices[-1] = np.arange(self.length-batch_size, self.length)
        
        return slices


    # def get_slice(self, index):
    #     items, n_node, A_in, A_out, alias_inputs = [], [], [], [], []
    #     for u_input in self.inputs[index]:
    #         n_node.append(len(np.unique(u_input)))
    #     max_n_node = np.max(n_node)
        
    #     for u_input in self.inputs[index]:
    #         node = np.unique(u_input)
    #         items.append(node.tolist() + (max_n_node - len(node)) * [0])
    #         u_A = np.zeros((max_n_node, max_n_node))
    #         for i in np.arange(len(u_input) - 1):
    #             if u_input[i + 1] == 0:
    #                 break
    #             u = np.where(node == u_input[i])[0][0]
    #             v = np.where(node == u_input[i + 1])[0][0]
    #             u_A[u][v] =1

    #         u_sum_in = np.sum(u_A, 0)
    #         u_sum_in[np.where(u_sum_in == 0)] = 1
    #         u_A_in = np.divide(u_A, u_sum_in)
    #         u_A_in = u_A_in + np.eye(u_A_in.shape[0])
    #         u_sum_out = np.sum(u_A, 1)
    #         u_sum_out[np.where(u_sum_out == 0)] = 1
    #         u_A_out = np.divide(u_A.transpose(), u_sum_out)
    #         u_A_out = u_A_out + np.eye(u_A_out.shape[0])

    #         A_in.append(u_A_in)
    #         A_out.append(u_A_out)
    #         alias_inputs.append([np.where(node == i)[0][0] for i in u_input])
    #     return A_in, A_out, alias_inputs, items, self.mask[index], self.targets[index]

    def get_slice(self, index):
        items, n_node, alias_inputs, = [], [], []
        for u_input in self.inputs[index]:
            #node = np.unique(u_input)
            #items.append(node.tolist() + (max_n_node - len(node)) * [0])
            alias_inputs.append([np.where(u_input == i)[0][0] for i in u_input])
        
        return alias_inputs, self.inputs[index], self.mask[index], self.targets[index]
    