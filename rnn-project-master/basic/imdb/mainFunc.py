from layers.rnn import *
from common.data import *
import h5py
import time


def pred_error(model, prepare_data, data, iterator):
    valid_err = 0
    for _, valid_index in iterator:
        x, mask, y = prepare_data([data[0][t] for t in valid_index],numpy.array(data[1])[valid_index],maxlen=None)

        print x.shape
        [v_error,h] = model.error(x, mask, y)
        valid_err += v_error
    valid_err /= len(iterator)

    return valid_err,h


""" params for RNN model
"""
n_words = 100000
in_size = 128
out_size = 2
hidden_size = [128]  # this parameter must be a list
maxlen = None
""" load the data
"""
print 'Loading data ...'
train_set, valid_set, test_set = load_imdb(n_words=n_words,
                                           valid_portion=0.1,
                                           maxlen=maxlen)
""" optimization
"""
batch_size = 256
valid_batch_size = 64

print 'optimization ...'
valid_shuffle= get_minibatches_idx(len(valid_set[0]), valid_batch_size,shuffle=True)
test_shuffle = get_minibatches_idx(len(test_set[0]), valid_batch_size)

print len(train_set);
"""
print zip(range(len(minibatches)), minibatches)
print (valid_shuffle[1])
valid_dat = valid_set[0]
print len(valid_dat[0])
"""

""" params for training
"""
patience = 10
max_epochs = 50
dispFreq = 10
lr = 0.001
validFreq = 20
saveFreq = 1000
dataset = 'imdb'
test_size = 500
saveto = 'models/model.h5'




print('%d train examples' % len(train_set[0]))
print('%d valid examples' % len(valid_set[0]))
print('%d test examples' % len(test_set[0]))

#
print 'Building the model ...'
# n_words is the max work
# in_size is the input size of the RNN input
# out_size is the output size of the RNN output
# hidden_size is the size of the hidden layer
model = RNN(n_words=n_words, in_size=in_size,
            out_size=out_size, hidden_size=hidden_size)

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

try:
    for eidx in range(max_epochs):
        n_samples = 0
        # shuffle the train data
        train_shuffle = get_minibatches_idx(len(train_set[0]), batch_size, shuffle=True)
        len(train_shuffle)
        for _, train_index in train_shuffle:
            uidx += 1
            x = [train_set[0][t] for t in train_index]
            y = [train_set[1][t] for t in train_index]
            x, mask, y = prepare_data(x, y)
            print x.shape
            print len(y)
            n_samples += x.shape[1]

            [cost,h] = model.train(x, mask, y, lr)
            if numpy.isnan(cost) or numpy.isinf(cost):
                print('bad cost detected: ', cost)
                break

            if numpy.mod(uidx, dispFreq) == 0:
                print('Epoch%d, Update%d, Cost%.6f' % (eidx, uidx, cost))

            if saveto and numpy.mod(uidx, saveFreq) == 0:
                print('Saving...')
                model.save_to_file(file_name=saveto, file_index=uidx)

            if numpy.mod(uidx, validFreq) == 0:
                train_err,h = pred_error(model, prepare_data, train_set, train_shuffle)
                valid_err,h = pred_error(model, prepare_data, valid_set, valid_shuffle)
                test_err,h = pred_error(model, prepare_data, test_set, test_shuffle)
                print 'music'
                file = h5py.File('music'+str(uidx)+'.h5','w')
                file.create_dataset('simi_matrix', data = h)
                print validFreq
                file.close()
                print(('Train ', train_err, 'Valid ', valid_err,
                       'Test ', test_err))


except KeyboardInterrupt:
    print('Training interupted ...')

end_time = time.time()
print('The code run for %d epochs, with %f sec/epochs' % (
    (eidx + 1), (end_time - start_time) / (1. * (eidx + 1))))