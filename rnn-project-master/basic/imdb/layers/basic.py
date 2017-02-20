import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import sys
sys.path.append('..')
from common.utils import *



class BasicLayer(object):
    def __init__(self, rng, layer_id, shape, X, mask,
                 use_noise=1, p=0.5):
        """
        Basic RNN with dropout

        Parameters
        ----------
        :param rng: can be generated as numpy.random.seed(123)

        :type layer_id: str
        :param layer_id: id of this layer

        :type shape: tuple
        :param shape: (in_size, out_size) where
                      in_size is the input dimension
                      out_size is the hidden units' dimension

        :type X: a 3D or 2D variable, mostly a 3D one
        :param X: model inputs

        :type mask: theano variable
        :param mask: model inputs

        :type use_noise: theano variable
        :param use_noise: whether dropout is random

        :type p: float
        :param p: dropout ratio
        """
        prefix = 'Basic' + layer_id
        self.in_size, self.hid_size,self.hid_pos_size = shape

        # weights for input
        self.W = init_weights(shape=(self.in_size, self.hid_size),
                              name=prefix + '#W')
        # weights for hidden states
        self.U = init_weights(shape=(self.hid_size, self.hid_size),
                              name=prefix + '#U')

        # weights for part hidden states


        if self.hid_pos_size != self.hid_size:
            self.U_LU = init_weights(shape=(self.hid_pos_size, self.hid_pos_size), name=prefix + '#ULU')
            self.U_LD = init_weights(shape=(self.hid_pos_size,self.hid_size - self.hid_pos_size),name=prefix + '#URU')
            self.U_RU = init_weights(shape=(self.hid_size-self.hid_pos_size,self.hid_pos_size),name=prefix + '#ULD')

        # bias
        self.b = init_bias(size=self.hid_size, name=prefix + '#b')

        self.X = X
        self.mask = mask

        nsteps = X.shape[0]
        if X.ndim == 3:
            n_samples = X.shape[1]
        else:
            n_samples = 1

        assert mask is not None

        def _slice(x, n, dim):
            if x.ndim == 3:
                return x[:, :, n * dim: (n + 1) * dim]
            return x[:, n * dim: (n + 1) * dim]

        def _step(x_t, m_t, h_tm1):
            """
            This function computes one step evolution in LSTM

            Parameters
            ----------
            :type m_t: (n_samples, )
            :param m_t: mask

            :type x_t: (n_samples, in_size)
            :param x_t: input at time t

            :type h_tm1: (n_samples, hid_size)
            :param h_tm1: hidden state at time (t - 1)
            """
            # h_t with size (n_samples, hid_size)

            if self.hid_pos_size != self.hid_size:
                h_pos = T.dot(h_tm1[:,0:self.hid_pos_size], self.U_LU) - T.dot(h_tm1[:,self.hid_pos_size:self.hid_size],T.nnet.relu(self.U_RU))
                h_neg = T.dot(h_tm1[:,0:self.hid_pos_size],T.nnet.relu(self.U_LD))
                preact = T.dot(x_t, self.W) + T.concatenate((h_pos, h_neg), axis=1) + self.b
            else:
                h_pos = T.dot(h_tm1, self.U)
                preact = T.dot(x_t, self.W) + h_pos + self.b
                print  self.hid_size,self.hid_pos_size
            #`preact = T.dot(x_t, self.W) + T.dot(h_tm1, self.U) + self.b

            h_t = T.nnet.relu(preact)
            # consider the mask
            h_t = m_t[:, None] * h_t + (1. - m_t)[:, None] * h_tm1

            return h_t

        h, updates = theano.scan(fn=_step,
                                 sequences=[self.X, self.mask],
                                 outputs_info=[T.alloc(floatX(0.),
                                                       n_samples,
                                                       self.hid_size)])
        # h here is of size (t, n_samples, hid_size)
        if p > 0:
            trng = RandomStreams(rng.randint(999999))
            drop_mask = trng.binomial(size=h.shape, n=1,
                                      p=(1 - p), dtype=theano.config.floatX)
            self.activation = T.switch(T.eq(use_noise, 1), h * drop_mask, h * (1 - p))
        else:
            self.activation = h
        #self.params = [self.W, self.U, self.b]

        if self.hid_pos_size != self.hid_size:
            self.params = [self.W, self.U_LD,self.U_LU,self.U_RU, self.b]
        else:
            self.params = [self.W,self.U,self.b]