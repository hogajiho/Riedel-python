# Riedel2013-python
#
# Based on python library theano_bpr
#
#
# Added neighborhood model
# by Jiho Kim (hogajiho@kaist.ac.kr)
# in KAIST SWRC lab

import theano, numpy
import theano.tensor as T
import time
import sys
from collections import defaultdict

class BPR(object):

    def __init__(self, rank, n_users, n_items, model_name = "f", n_entities, lambda_u = 0.0025, lambda_i = 0.0025, \
      lambda_j = 0.00025, lambda_bias = 0.0, learning_rate = 0.05):
        """
          Creates a new object for training and testing a Bayesian
          Personalised Ranking (BPR) Matrix Factorisation 
          model, as described by Rendle et al. in:

            http://arxiv.org/abs/1205.2618

          This model tries to predict a ranking of items for each user
          from a viewing history.  
          It's also used in a variety of other use-cases, such
          as matrix completion, link prediction and tag recommendation.

          `rank` is the number of latent features in the matrix
          factorisation model.

          `n_users` is the number of users and `n_items` is the
          number of items.

          The regularisation parameters can be overridden using
          `lambda_u`, `lambda_i` and `lambda_j`. They correspond
          to each three types of updates.

          The learning rate can be overridden using `learning_rate`.

          This object uses the Theano library for training the model, meaning
          it can run on a GPU through CUDA. To make sure your Theano
          install is using the GPU, see:

            http://deeplearning.net/software/theano/tutorial/using_gpu.html

          When running on CPU, we recommend using OpenBLAS.

            http://www.openblas.net/

          Example use (10 latent dimensions, 100 users, 50 items) for
          training:

          >>> from theano_bpr import BPR
          >>> bpr = BPR(10, 100, 50) 
          >>> from numpy.random import randint
          >>> train_data = zip(randint(100, size=1000), randint(50, size=1000))
          >>> bpr.train(train_data)

          This object also has a method for testing, which will return
          the Area Under Curve for a test set.

          >>> test_data = zip(randint(100, size=1000), randint(50, size=1000))
          >>> bpr.test(test_data)

          (This should give an AUC of around 0.5 as the training and
          testing set are chosen at random)
        """
        self._rank = rank
        self._n_users = n_users
        self._n_items = n_items
        self._model_name = model_name
        self._lambda_u = lambda_u
        self._lambda_i = lambda_i
        self._lambda_j = lambda_j
        self._lambda_bias = lambda_bias
        self._learning_rate = learning_rate
        self._train_users = set()
        self._train_items = set()
        self._train_dict = {}
        self._tuple_entity_match = {}
        self._configure_theano()

    def _configure_theano(self):
        """
          Configures Theano to run in fast mode
          and using 32-bit floats. 
        """
        theano.config.mode = 'FAST_RUN'
        theano.config.floatX = 'float32'

    def _generate_train_model_function_f(self):
        u = T.lvector('u')
        i = T.lvector('i')
        j = T.lvector('j')

        self.W = theano.shared(numpy.random.random((self._n_users, self._rank)).astype('float32'), name='W')
        self.H = theano.shared(numpy.random.random((self._n_items, self._rank)).astype('float32'), name='H')

        self.B = theano.shared(numpy.zeros(self._n_items).astype('float32'), name='B')

        x_ui = T.dot(self.W[u], self.H[i].T).diagonal()
        x_uj = T.dot(self.W[u], self.H[j].T).diagonal()

        x_uij = self.B[i] - self.B[j] + x_ui - x_uj

        obj = T.sum(T.log(T.nnet.sigmoid(x_uij)) - self._lambda_u * (self.W[u] ** 2).sum(axis=1) \
          - self._lambda_i * (self.H[i] ** 2).sum(axis=1) - self._lambda_j * (self.H[j] ** 2).sum(axis=1) \
          - self._lambda_bias * (self.B[i] ** 2 + self.B[j] ** 2))
        cost = - obj

        g_cost_W = T.grad(cost=cost, wrt=self.W)
        g_cost_H = T.grad(cost=cost, wrt=self.H)
        g_cost_B = T.grad(cost=cost, wrt=self.B)

        updates = [ (self.W, self.W - self._learning_rate * g_cost_W), \
        (self.H, self.H - self._learning_rate * g_cost_H), (self.B, self.B - self._learning_rate * g_cost_B) ]

        self.train_model = theano.function(inputs=[u, i, j], outputs=cost, updates=updates)

    def _generate_train_model_function_n(self):
        u = T.lvector('u')
        i = T.lvector('i')
        j = T.lvector('j')

        self.RW = theano.shared(numpy.random.random((self._n_items, self._n_items)).astype('float32'), name='RW')
        self.B = theano.shared(numpy.zeros(self._n_items).astype('float32'), name='B')
        temp = numpy.zeros((self._n_users, self._n_items))
        for key in self._train_dict.keys():
          for item in self._train_dict[key]:
            temp[key][item] = 1
        self.TR = theano.shared(temp.astype('float32'), name='TR')

        x_ui = T.dot(self.RW[i], self.TR[u].T).diagonal()
        x_uj = T.dot(self.RW[j], self.TR[u].T).diagonal()

        x_uij = self.B[i] - self.B[j] + x_ui - x_uj

        obj = T.sum(T.log(T.nnet.sigmoid(x_uij)) - self._lambda_i * (self.RW[i] ** 2).sum(axis=1) \
          - self._lambda_j * (self.RW[j] ** 2).sum(axis=1) - self._lambda_bias * (self.B[i] ** 2 + self.B[j] ** 2))
        cost = - obj

        g_cost_RW = T.grad(cost=cost, wrt=self.RW)
        g_cost_B = T.grad(cost=cost, wrt=self.B)

        updates = [(self.RW, self.RW - self._learning_rate * g_cost_RW), (self.B, self.B - self._learning_rate * g_cost_B)]

        self.train_model = theano.function(inputs=[u, i, j], outputs=cost, updates=updates)

    def _generate_train_model_function_nf(self):
        u = T.lvector('u')
        i = T.lvector('i')
        j = T.lvector('j')

        self.W = theano.shared(numpy.random.random((self._n_users, self._rank)).astype('float32'), name='W')
        self.H = theano.shared(numpy.random.random((self._n_items, self._rank)).astype('float32'), name='H')
        self.RW = theano.shared(numpy.random.random((self._n_items, self._n_items)).astype('float32'), name='RW')
        temp = numpy.zeros((self._n_users, self._n_items))
        for key in self._train_dict.keys():
          for item in self._train_dict[key]:
            temp[key][item] = 1
        self.TR = theano.shared(temp.astype('float32'), name='TR')

        self.B = theano.shared(numpy.zeros(self._n_items).astype('float32'), name='B')

        x_ui = T.dot(self.W[u], self.H[i].T).diagonal() + T.dot(self.RW[i], self.TR[u].T).diagonal()
        x_uj = T.dot(self.W[u], self.H[j].T).diagonal() + T.dot(self.RW[j], self.TR[u].T).diagonal()

        x_uij = self.B[i] - self.B[j] + x_ui - x_uj

        obj = T.sum(T.log(T.nnet.sigmoid(x_uij)) - self._lambda_i * (self.RW[i] ** 2).sum(axis=1) \
          - self._lambda_j * (self.RW[j] ** 2).sum(axis=1)  - self._lambda_u * (self.W[u] ** 2).sum(axis=1) \
          - self._lambda_i * (self.H[i] ** 2).sum(axis=1) - self._lambda_j * (self.H[j] ** 2).sum(axis=1) \
          - self._lambda_bias * (self.B[i] ** 2 + self.B[j] ** 2))
        cost = - obj

        g_cost_W = T.grad(cost=cost, wrt=self.W)
        g_cost_H = T.grad(cost=cost, wrt=self.H)
        g_cost_RW = T.grad(cost=cost, wrt=self.RW)
        g_cost_B = T.grad(cost=cost, wrt=self.B)

        updates = [ (self.W, self.W - self._learning_rate * g_cost_W), (self.H, self.H - self._learning_rate * g_cost_H), (self.RW, self.RW - self._learning_rate * g_cost_RW), (self.B, self.B - self._learning_rate * g_cost_B) ]

        self.train_model = theano.function(inputs=[u, i, j], outputs=cost, updates=updates)

    def _generate_train_model_function_e(self):
        u = T.lvector('u')
        i = T.lvector('i')
        j = T.lvector('j')

        self.RW = theano.shared(numpy.random.random((self._n_items, self._n_items)).astype('float32'), name='RW')
        self.B = theano.shared(numpy.zeros(self._n_items).astype('float32'), name='B')
        temp = numpy.zeros((self._n_users, self._n_items))
        for key in self._train_dict.keys():
          for item in self._train_dict[key]:
            temp[key][item] = 1
        self.TR = theano.shared(temp.astype('float32'), name='TR')

        x_ui = T.dot(self.RW[i], self.TR[u].T).diagonal()
        x_uj = T.dot(self.RW[j], self.TR[u].T).diagonal()

        x_uij = self.B[i] - self.B[j] + x_ui - x_uj

        obj = T.sum(T.log(T.nnet.sigmoid(x_uij)) - self._lambda_i * (self.RW[i] ** 2).sum(axis=1) \
          - self._lambda_j * (self.RW[j] ** 2).sum(axis=1) - self._lambda_bias * (self.B[i] ** 2 + self.B[j] ** 2))
        cost = - obj

        g_cost_RW = T.grad(cost=cost, wrt=self.RW)
        g_cost_B = T.grad(cost=cost, wrt=self.B)

        updates = [ (self.RW, self.RW - self._learning_rate * g_cost_RW), (self.B, self.B - self._learning_rate * g_cost_B) ]

        self.train_model = theano.function(inputs=[u, i, j], outputs=cost, updates=updates)

    def _generate_train_model_function_nfe(self):  

    def train(self, train_data, tuple_entity_match, epochs=30, batch_size=1000):
        """
          Trains the BPR Matrix Factorisation model using Stochastic
          Gradient Descent and minibatches over `train_data`.

          `train_data` is an array of (user_index, item_index) tuples.

          We first create a set of random samples from `train_data` for 
          training, of size `epochs` * size of `train_data`.

          We then iterate through the resulting training samples by
          batches of length `batch_size`, and run one iteration of gradient
          descent for the batch.
        """
        if len(train_data) < batch_size:
            sys.stderr.write("WARNING: Batch size is greater than \number of training samples, switching to a batch size of %s\n" % str(len(train_data)))
            batch_size = len(train_data)
        self._train_dict, self._train_users, self._train_items = self._data_to_dict(train_data)

        if self._model_name == "f":
          self._generate_train_model_function_f()
        elif self._model_name == "n":
          self._generate_train_model_function_n()
        elif self._model_name == "nf":
          self._generate_train_model_function_nf()

        n_sgd_samples = len(train_data) * epochs
        sgd_users, sgd_pos_items, sgd_neg_items = self._uniform_user_sampling(n_sgd_samples)
        z = 0
        t2 = t1 = t0 = time.time()
        while (z+1)*batch_size < n_sgd_samples:
            self.train_model(
                sgd_users[z*batch_size: (z+1)*batch_size],
                sgd_pos_items[z*batch_size: (z+1)*batch_size],
                sgd_neg_items[z*batch_size: (z+1)*batch_size]
            )
            z += 1
            t2 = time.time()
            sys.stderr.write("\rProcessed %s ( %.2f%% ) in %.4f seconds" %(str(z*batch_size), \
              100.0 * float(z*batch_size)/n_sgd_samples, t2 - t1))
            sys.stderr.flush()
            t1 = t2
        if n_sgd_samples > 0:
            sys.stderr.write("\nTotal training time %.2f seconds; %e per sample\n" % (t2 - t0, (t2 - t0)/n_sgd_samples))
            sys.stderr.flush()

    def _uniform_user_sampling(self, n_samples):
        """
          Creates `n_samples` random samples from training data for performing Stochastic
          Gradient Descent. We start by uniformly sampling users, 
          and then sample a positive and a negative item for each 
          user sample.
        """
        sys.stderr.write("Generating %s random training samples\n" % str(n_samples))
        sgd_users = numpy.array(list(self._train_users))[numpy.random.randint(len(list(self._train_users)), size=n_samples)]
        sgd_pos_items, sgd_neg_items = [], []
        for sgd_user in sgd_users:
            pos_item = self._train_dict[sgd_user][numpy.random.randint(len(self._train_dict[sgd_user]))]
            sgd_pos_items.append(pos_item)
            neg_item = numpy.random.randint(self._n_items)
            while neg_item in self._train_dict[sgd_user]:
                neg_item = numpy.random.randint(self._n_items)
            sgd_neg_items.append(neg_item)
        return sgd_users, sgd_pos_items, sgd_neg_items

    def predictions(self, user_index):
        """
          Computes item predictions for `user_index`.
          Returns an array of prediction values for each item
          in the dataset.
        """
        if self._model_name == "f":
          w = self.W.get_value()
          h = self.H.get_value()
          b = self.B.get_value()
          user_vector = w[user_index,:]
          return_vector = user_vector.dot(h.T) + b
          #print return_vector
          return return_vector
        elif self._model_name == "n":
          rw = self.RW.get_value()
          #print rw.shape
          tr = self.TR.get_value()
          #print tr.shape
          b = self.B.get_value()
          return_vector = numpy.zeros(self._n_items)
          for i in range(self._n_items):
            return_vector[i] = tr[user_index, :].dot(rw[i,:].T) + b[i]
          #print return_vector
          return return_vector
        elif self._model_name == "nf":
          w = self.W.get_value()
          h = self.H.get_value()
          b = self.B.get_value()
          rw = self.RW.get_value()
          tr = self.TR.get_value()
          return_vector = numpy.zeros(self._n_items)
          for i in range(self._n_items):
            return_vector[i] = tr[user_index, :].dot(rw[i,:].T)
          user_vector = w[user_index,:]
          return_vector += user_vector.dot(h.T) + b
          return return_vector



    def prediction(self, user_index, item_index):
        """
          Predicts the preference of a given `user_index`
          for a gven `item_index`.
        """
        return self.predictions(user_index)[item_index]

    def top_predictions(self, user_index, topn=10):
        """
          Returns the item indices of the top predictions
          for `user_index`. The number of predictions to return
          can be set via `topn`.
          This won't return any of the items associated with `user_index`
          in the training set.
        """
        return [ 
            item_index for item_index in numpy.argsort(self.predictions(user_index)) 
            if item_index not in self._train_dict[user_index]
        ][::-1][:topn]

    def test(self, test_data):
        """
          Computes the Area Under Curve (AUC) on `test_data`.

          `test_data` is an array of (user_index, item_index) tuples.

          During this computation we ignore users and items
          that didn't appear in the training data, to allow
          for non-overlapping training and testing sets.
        """
        test_dict, test_users, test_items = self._data_to_dict(test_data)
        auc_values = []
        z = 0
        for user in test_dict.keys():
            if user in self._train_users:
                auc_for_user = 0.0
                n = 0
                predictions = self.predictions(user)
                for pos_item in test_dict[user]:
                    if pos_item in self._train_items:
                        for neg_item in self._train_items:
                            if neg_item not in test_dict[user] and neg_item not in self._train_dict[user]:
                                n += 1
                                if predictions[pos_item] > predictions[neg_item]:
                                    auc_for_user += 1
                if n > 0:
                    auc_for_user /= n
                    auc_values.append(auc_for_user)
                z += 1
                if z % 100 == 0 and len(auc_values) > 0:
                    sys.stderr.write("\rCurrent AUC mean (%s samples): %0.5f" % (str(z), numpy.mean(auc_values)))
                    sys.stderr.flush()
        sys.stderr.write("\n")
        sys.stderr.flush()
        return numpy.mean(auc_values)

    def _data_to_dict(self, data):
        data_dict = defaultdict(list)
        items = set()
        for (user, item) in data:
            data_dict[user].append(item)
            items.add(item)
        return data_dict, set(data_dict.keys()), items
