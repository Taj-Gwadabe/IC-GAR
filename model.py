import tensorflow as tf
import math
import numpy as np
tf.set_random_seed(42)
np.random.seed(42)


class Model(object):
    def __init__(self, hidden_size=100, out_size=100, batch_size=100, nonhybrid=False):
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.batch_size = batch_size
        self.mask = tf.compat.v1.placeholder(dtype=tf.float32)
        self.alias = tf.compat.v1.placeholder(dtype=tf.int32) 
        self.item = tf.compat.v1.placeholder(dtype=tf.int32)  
        self.tar = tf.compat.v1.placeholder(dtype=tf.int32)
        self.nonhybrid = nonhybrid
        self.stdv = 1.0 / math.sqrt(self.hidden_size)

        
        self.nasr_w1 = tf.Variable(tf.random.uniform((self.out_size, self.out_size), -self.stdv, self.stdv), name='nasr_w1', dtype=tf.float32)
        self.nasr_w2 = tf.Variable(tf.random.uniform((self.out_size, self.out_size), -self.stdv, self.stdv), name='nasr_w2', dtype=tf.float32)
        self.nasr_w3 = tf.Variable(tf.random.uniform((self.out_size, self.out_size), -self.stdv, self.stdv), name='nasr_w3', dtype=tf.float32)
        self.nasr_v = tf.Variable(tf.random.uniform((1, self.out_size), -self.stdv, self.stdv), name='nasr_v', dtype=tf.float32)   
        self.nasr_b = tf.Variable(tf.random.uniform((self.out_size, ), -self.stdv, self.stdv), name='nasr_b', dtype=tf.float32)
       
    def forward(self, re_embedding, embed, train=True):
        rm = tf.reduce_sum(self.mask, 1)
        last_id = tf.gather_nd(self.alias, tf.stack([tf.range(self.batch_size), tf.cast(rm, tf.int32)-1], axis=1))
        last_h = tf.gather_nd(re_embedding, tf.stack([tf.range(self.batch_size), last_id], axis=1))
        seq_h = tf.stack([tf.nn.embedding_lookup(re_embedding[i], self.alias[i]) for i in range(self.batch_size)],axis=0) #batch_size*T*d

        temp_item = self.item-1
        last_emb = []
        for j in range(self.batch_size):
            row = temp_item[j]
            temp = tf.reduce_mean(tf.gather(embed, tf.where(tf.not_equal(row, -1))), 0)
            last_emb.append(temp)
        
        last = tf.reshape(last_h, [self.batch_size, 1, -1])
        seq = tf.matmul(tf.reshape(seq_h, [-1, self.out_size]), self.nasr_w2)
        last_emb = tf.reshape(last_emb, [-1, self.out_size])
        last_embed = tf.matmul(last_emb, self.nasr_w3)
        
        b = self.embedding[1:]
    
        m = tf.nn.sigmoid(last + tf.reshape(seq, [self.batch_size, -1, self.out_size]) + self.nasr_b)
        coef = tf.matmul(tf.reshape(m, [-1, self.out_size]), self.nasr_v, transpose_b=True) * tf.reshape(self.mask, [-1, 1])
        glb_emb = tf.reshape(tf.reduce_sum(tf.reshape(coef, [self.batch_size, -1, 1]) * seq_h, 1),[-1, self.out_size])
        gate_embed = glb_emb + last_embed
        last_h = tf.reshape(last_h, [-1, self.out_size])
        ma = tf.multiply(gate_embed, last_h)
        logits = tf.matmul(ma, b, transpose_b=True)
      
        loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.tar - 1, logits=logits))
        self.vars = tf.compat.v1.trainable_variables()
        if train:
            lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in self.vars if v.name not
                               in ['bias', 'gamma', 'b', 'g', 'beta']]) * self.L2
            loss = loss + lossL2
        return loss, logits

    def run(self, fetches, tar, item, alias, mask):
        return self.sess.run(fetches, feed_dict={self.tar: tar, self.item: item,
                                                     self.alias: alias, self.mask: mask})

class GRASER(Model):
    def __init__(self, adj, hidden_size=100, out_size=100, batch_size=100,n_node=None, lr=None, l2=None, step=1, 
                                 decay=None, lr_dc=0.1, nonhybrid=False):
        super(GRASER,self).__init__(hidden_size, out_size, batch_size, nonhybrid)
    
        self.embedding = tf.Variable(tf.random.uniform((n_node, hidden_size), -self.stdv, self.stdv), name='embedding', dtype=tf.float32)
        self.n_node = n_node
        self.adj = adj
        self.L2 = l2
        self.step = step
        self.n_fold = 10
        self.mess_dropout = [0.2,0.2]
        self.node_dropout = [0.4]
        self.nonhybrid = nonhybrid
        self.weight_size = [100,100]
        self.n_layers = len(self.weight_size)
        self.weights = self._init_weights()
   
        with tf.compat.v1.variable_scope('graser_model', reuse=None):
            embed_out=self.gcn_embed(self.adj)
            gru_out = self.gru()
            self.loss_train, _ = self.forward(gru_out, embed_out)
        
        with tf.compat.v1.variable_scope('graser_model', reuse=True):
            embed_out = self.gcn_embed(self.adj)
            gru_out = self.gru()
            self.loss_test, self.score_test = self.forward(gru_out, embed_out, train=False)
        
        self.global_step = tf.Variable(0)
        self.learning_rate = tf.compat.v1.train.exponential_decay(lr, global_step=self.global_step, decay_steps=decay,
                                                        decay_rate=lr_dc, staircase=True)
        self.opt = tf.compat.v1.train.AdamOptimizer(self.learning_rate).minimize(self.loss_train, global_step=self.global_step)
        gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8)
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.compat.v1.Session(config=config)
        self.sess.run(tf.global_variables_initializer())

    def _init_weights(self):
        all_weigths = dict()
        initializer = tf.random_uniform_initializer(-self.stdv, self.stdv)
        self.weight_size_list = [self.hidden_size] + self.weight_size
        for k in range(self.n_layers):
            all_weigths['W_gc_%d' %k] = tf.Variable(
                initializer([self.weight_size_list[k], self.weight_size_list[k+1]]), name='W_gc_%d' %k)
            all_weigths['b_gc_%d' %k] = tf.Variable(
                initializer([self.weight_size_list[k+1]]), name='b_gc_%d' %k)
            all_weigths['W_concat'] = tf.Variable(
                initializer([(1+self.n_layers)*self.weight_size_list[k], self.weight_size_list[k+1]]), name='W_concat')
            all_weigths['b_concat'] = tf.Variable(
                initializer([self.weight_size_list[k+1]]), name='b_concat')
        return all_weigths
   
    def _split_A_hat_with_dropout(self, X):
        A_fold_hat = []
        fold_len = (self.n_node) // self.n_fold
        for i in range(self.n_fold):
            start = i*fold_len
            if i==self.n_fold-1:
                end = self.n_node
            else:
                end = (i+1)*fold_len
            temp = self._convert_to_sp_mat(X[start:end])
            n_nonzero = X[start:end].count_nonzero()
            A_fold_hat.append(self._dropout_sparse(temp, 1- self.node_dropout[0], n_nonzero))
        return A_fold_hat

    def _dropout_sparse(self, X, keep_prob, n_nonzero):
        noise_shape = [n_nonzero]
        random_tensor = keep_prob
        random_tensor += tf.random_uniform(noise_shape)
        dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
        pre_out = tf.sparse_retain(X, dropout_mask)
        return pre_out*tf.div(1., keep_prob)
    
    def _convert_to_sp_mat(self, X):
        coo = X.tocoo().astype(np.float32)
        indices = np.mat([coo.row, coo.col]).transpose()
        return tf.SparseTensor(indices, coo.data, coo.shape)

    def gcn_embed(self, A):
        A_fold_hat = self._split_A_hat_with_dropout(A)
        ego_embeddings = self.embedding
        all_embeddings = [ego_embeddings]
        for k in range(0, self.n_layers):
            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(tf.sparse.sparse_dense_matmul(A_fold_hat[f], ego_embeddings))
            embeddings = tf.concat(temp_embed, 0)
            embeddings = tf.nn.leaky_relu(
                tf.matmul(embeddings, self.weights['W_gc_%d' %k]) + self.weights['b_gc_%d' %k])
            embeddings = tf.nn.dropout(embeddings, 1-self.mess_dropout[k])
            all_embeddings += [embeddings]
        all_embeddings = tf.concat(all_embeddings, 1)
        all_embeddings = tf.nn.leaky_relu(
         tf.matmul(all_embeddings, self.weights['W_concat']) + self.weights['b_concat'])
        return all_embeddings

    def gru(self):
        state = tf.nn.embedding_lookup(self.embedding, self.item)
        with tf.compat.v1.variable_scope('gru'):
            cell1 = tf.nn.rnn_cell.GRUCell(self.out_size,activation=tf.nn.tanh)
            for i in range(self.step):
                state_output, fin_state = \
                    tf.nn.dynamic_rnn(cell1, tf.expand_dims(tf.reshape(state, [-1, self.out_size]), axis=1),
                               initial_state=tf.reshape(state, [-1, self.out_size]), dtype = tf.float32)        
        return tf.reshape(fin_state, [self.batch_size, -1, self.out_size])