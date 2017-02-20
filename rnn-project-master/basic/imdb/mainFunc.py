from layers.rnn import *
from common.data import *

import time
import os
from collections import Counter


def pred_error(model, prepare_data, data, iterator):
    valid_err = 0
    for _, valid_index in iterator:
        x, mask, y = prepare_data([data[0][t] for t in valid_index],
                                  numpy.array(data[1])[valid_index],
                                  maxlen=None)
        #add by kalen:q:
        err,h = model.error(x, mask, y)
        valid_err += err
    valid_err /= len(iterator)

    return valid_err,h


""" params for RNN model
"""
n_words = 100000
in_size = 128
out_size = 2
hidden_size = [64]  # this parameter must be a list
hidden_pos_size0 = hidden_size[0]
for i in range(0,10):
    print 'Building the model ...'
    print 'hidden neg size is '
    hidden_neg_size = i*4
    new_hidden_size = [hidden_size[0]-hidden_neg_size]
    print hidden_neg_size
    print 'hidden pos size is'
    print new_hidden_size
    model = RNN(n_words=n_words, in_size=in_size,
                out_size=out_size, hidden_size=new_hidden_size,hidden_neg_size = hidden_neg_size,cell='brnn')

    """ params for training
    """
    patience = 10
    max_epochs = 6
    dispFreq = 10
    lr = 0.001
    validFreq = 100
    saveFreq = 100000000
    maxlen = None
    batch_size = 128
    valid_batch_size = 64
    dataset = 'imdb'
    test_size = 500
    saveto = 'models/model.h5'

    """ load the data
    """
    print 'Loading data ...'
    train_set, valid_set, test_set = load_imdb(n_words=n_words,
                                               valid_portion=0.1,
                                               maxlen=maxlen)

    """ optimization
    """
    print 'optimization ...'
    valid_shuffle = get_minibatches_idx(len(valid_set[0]), valid_batch_size)
    test_shuffle = get_minibatches_idx(len(test_set[0]), valid_batch_size)

    print('%d train examples' % len(train_set[0]))
    print('%d valid examples' % len(valid_set[0]))
    print('%d test examples' % len(test_set[0]))

    history_errs = []
    best_p = None
    bad_counter = 0

    if validFreq == -1:
        validFreq = len(train_set[0]) // batch_size
    if saveFreq == -1:
        saveFreq = len(train_set[0]) // batch_size

    uidx = 0  # the number of update done
    estop = False  # early stop
    start_time = time.time()

    train_error = numpy.zeros((20000))
    test_error = numpy.zeros((20000))
    valid_error = numpy.zeros((20000))
    once = True
    def show_param(model,index = -1):
        param = model.showparam()
        if index == -1:
            all_p = []
            for _,p in param:
                all_p.append(p.get_value())
            return all_p
        return param[index].get_value()
    try:
        for eidx in range(max_epochs):
            n_samples = 0
            train_shuffle = get_minibatches_idx(len(train_set[0]), batch_size, shuffle=True)

            for _, train_index in train_shuffle:
                uidx += 1
                x = [train_set[0][t] for t in train_index]
                y = [train_set[1][t] for t in train_index]
                x, mask, y = prepare_data(x, y)
                n_samples += x.shape[1]


                if once:
                    once = False
                    print x.shape
                    last,first = model.cacu_p_y_given_x(x,mask)
                    if numpy.isnan(last).any():
                        print numpy.isnan(last)
                    print x.shape
                    print last.shape
                    print numpy.count_nonzero(numpy.isnan(last))
                cost = model.train(x, mask, y, lr)
                last, first = model.cacu_p_y_given_x(x, mask)
                #print numpy.count_nonzero(numpy.isnan(last))
                #print numpy.count_nonzero(numpy.isnan(show_param(model,1)))
                if numpy.isnan(cost) or numpy.isinf(cost):
                    print('bad cost detected: ', cost)
                    break

                if numpy.mod(uidx, dispFreq) == 0:
                    print('Epoch%d, Update%d, Cost%.6f' % (eidx, uidx, cost))

                if saveto and numpy.mod(uidx, saveFreq) == 0:
                    print('Saving...')
                    model.save_to_file(file_name=saveto, file_index=uidx)
                if numpy.mod(uidx, validFreq) == 0:
                    train_err,h_train = pred_error(model, prepare_data, train_set, train_shuffle)
                    valid_err,h_valid = pred_error(model, prepare_data, valid_set, valid_shuffle)
                    test_err,h_test = pred_error(model, prepare_data, test_set, test_shuffle)
                    print h_train.shape
                    print h_test.shape
                    '''
                    hidden_filename = 'reluparam_'+str(uidx)+'_'+str(i)+'_'+str(hidden_size[0])+'.h5'
                    if os.path.exists(hidden_filename):
                        os.remove(hidden_filename)

                    f = h5py.File(hidden_filename)
                    f['train'] = h_train
                    f['valid'] = h_valid
                    f['test'] = h_test
                    f['param1'] = show_param(model, 1)
                    f['param2'] = show_param(model, 2)
                    f['param3'] = show_param(model, 3)
                    '''
                    print(('Train ', train_err, 'Valid ', valid_err,
                           'Test ', test_err))
                    train_error[uidx] = train_err
                    valid_error[uidx] = valid_err
                    test_error[uidx] = test_err
        f = h5py.File('error_all_neg_newmodel_' + str(i) + '.h5')
        f['train'] = train_error
        f['valid'] = valid_error
        f['test'] = test_error
    except KeyboardInterrupt:
        print('Training interupted ...')




end_time = time.time()
print('The code run for %d epochs, with %f sec/epochs' % (
    (eidx + 1), (end_time - start_time) / (1. * (eidx + 1))))
