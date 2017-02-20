#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import theano
import numpy as np
import theano.tensor as T

from convolutional_mlp import LeNetConvPoolLayer


class RCnnWithBpr(object):

    def __init__(self, n_basket,  n_hidden, n_vocabulary, n_embedding_dimension):
        # n_window to put together a few records of input training (may enhance the sense of sequence)
        # Such that x is an n_embedding_dimension * n_window-dimensional vector
        # The transformation matrix of the word vector (broadly, the property vector of the item)

        iscnn = False
        iscostplus = True
        nkerns = n_embedding_dimension  # Seemingly can not equal! . !! . . . . . First keep
        filter_shape = (nkerns, 1, n_basket, 1)
        # rng = np.random.RandomState(23455)
        rng = np.random.RandomState(23456)
        poolsize = (1, n_embedding_dimension)

        print "1. Neuron parameter construction ............",
        embedding = np.random.uniform(-0.5, 0.5, (n_vocabulary, n_embedding_dimension)).astype(theano.config.floatX)
        embedding[-1] = 0.
        self.embedding = theano.shared(
            value=embedding.astype(theano.config.floatX),
            name='embedding',
            borrow=True
        )

        # Simply defining -1 as an attribute does not seem right, but you can not think of a good way to change it.
        #  X by u
        self.u = theano.shared(
            value=np.random.uniform(-0.5, 0.5, (nkerns, n_hidden)).astype(theano.config.floatX),
            # This dimension should be modified like the vector dimension of the cnn output
            name='u',
            borrow=True
        )

        #  H by w
        self.w = theano.shared(
            value=np.random.uniform(-0.5, 0.5, (n_hidden, n_hidden)).astype(theano.config.floatX),
            name='w',
            borrow=True
        )

        self.hidden_lay0 = theano.shared(
            value=np.zeros(n_hidden, dtype=theano.config.floatX),
            name='hidden_lay0',
            borrow=True
        )

        fan_in = np.prod(filter_shape[1:])
        fan_out = (filter_shape[0] * np.prod(filter_shape[2:]) // np.prod(poolsize))

        # initialize weights with random weights
        W_bound = np.sqrt(6. / (fan_in + fan_out))
        self.w_cnn = theano.shared(np.asarray(rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                                              dtype=theano.config.floatX), borrow=True)
        b_values = np.zeros((filter_shape[0],), dtype=theano.config.floatX)

        self.b_cnn = theano.shared(value=b_values, borrow=True)

        input_item_id = T.lmatrix('input_item_id')   # The the input matrix
        input_size = T.lvector('input_size')
        neg_item_id = T.lmatrix('neg_item_id')
        x = self.embedding[input_item_id].reshape((input_item_id.shape[0], n_basket, n_embedding_dimension))
        x.name = 'x'
        # y = self.embedding[next_item_id].reshape((1, n_window * n_embedding_dimension))[0]
        neg = self.embedding[neg_item_id].reshape((neg_item_id.shape[0], n_basket, n_embedding_dimension))
        neg.name = 'neg'

        # After the embedding of the feature matrix through a cnn
        # . . . Note the convolution here first
        if iscnn:
            cnn_x = LeNetConvPoolLayer(
                rng,
                input=x.reshape((x.shape[0], 1, n_basket, n_embedding_dimension)),
                image_shape=(None, 1, n_basket,  n_embedding_dimension),
                # In fact image_shape almost no role in this variable, the first dimension casually write on the line
                filter_shape=filter_shape,
                W=self.w_cnn,
                b=self.b_cnn,
                poolsize=poolsize
            )
            cnn_x_output = cnn_x.output.flatten(2)
            self.param = (self.embedding, self.u, self.w, self.w_cnn, self.b_cnn)  # , self.v)
            self.name = ('embedding', 'u', 'w', 'w_cnn', 'b_cnn')
        else:
            def pooling_max(abasker_t, basket_size_t):
                pool_result_t = T.max(abasker_t[: basket_size_t], axis=0)
                return pool_result_t
            pool_result, _ = theano.scan(fn=pooling_max,
                                         sequences=[x.reshape((x.shape[0], n_basket, n_embedding_dimension)),
                                                    input_size])
            cnn_x = pool_result
            cnn_x_output = cnn_x.flatten(2)
            self.param = (self.embedding, self.u, self.w)
            self.name = ('embedding', 'u', 'w')

        print "done"

        print "2. Loss function construction ..............",

        def recurrence(x_t, h_tml):
            #  Defines the looping function
            h_t = T.nnet.sigmoid(T.dot(x_t, self.u) + T.dot(h_tml, self.w))
            return h_t

        h, _ = theano.scan(
            fn=recurrence,
            sequences=cnn_x_output,
            outputs_info=[self.hidden_lay0]
        )
        h.name = 'h'
        self.user_feature = h[-1, :]  #
        self.user_feature.name = 'user_feature'

        #  Loss function
        if iscostplus:
            def cla_cost(x_t, h_t):
                s_tt = T.dot((x[x_t+1][:input_size[x_t+1]] - neg[x_t+1][:input_size[x_t+1]]), h_t)
                s_t = T.sum(T.log(1 + T.exp(-s_tt)))
                return s_t
            s, _ = theano.scan(
                fn=cla_cost,
                sequences=[T.arange(x.shape[0]-1), h]
            )
            cost = T.sum(s)
        else:
            cost_temp = T.dot(x[-1][:input_size[-1]], h[-2]) - T.dot(neg[-1][:input_size[-1]], h[-2])
            cost = T.sum(T.log(1 + T.exp(-cost_temp)))

        print "done"

        print "3. Random gradient descending update formula ......",
        learning_rate = T.dscalar('learning_rate')
        lamda = T.dscalar('lamda')
        gradient = T.grad(cost, self.param)
        updates = [(p, p - learning_rate * (g + p * lamda)) for p, g in zip(self.param, gradient)]
        print "done"

        print "4. Predictive function definition ..............",
        y_pred = T.argsort(T.dot(self.embedding, self.user_feature)) # quel -6 fa prendere la top 5 in ordine crescente
        self.predict = theano.function(inputs=[input_item_id, input_size], outputs=y_pred)
        print "done"

        print "5. Training function definition ..............",
        self.train = theano.function(inputs=[input_item_id, neg_item_id, input_size, learning_rate, lamda],
                                     outputs=cost,
                                     updates=updates)
        print "done"

        print "6. Evaluation function definition ..............",
        self.evaluation_recall_6 = theano.function(inputs=[input_item_id, input_size], outputs=y_pred)
        print "done"

        self.normalize = theano.function(inputs=[],
                                         updates={self.embedding:\
                                         self.embedding/T.sqrt((self.embedding**2).sum(axis=1)).dimshuffle(0, 'x')*10})

    def init_hidden_layer(self):   # Seemingly useless
        n_out = self.hidden_lay0.shape[0]
        self.hidden_lay0.set_value(np.zeros(n_out, dtype=theano.config.floatX))

    def save_params(self, folder):
        for paramtemp, nametemp in zip(self.param, self.name):
            np.save(os.path.join(folder, nametemp + '.npy'), paramtemp.get_value())

    def load_params(self, folder):
        for paramtemp, nametemp in zip(self.param, self.name):
            paramtemp.set_value(np.load(os.path.join(folder, nametemp + '.npy')))
