'''
Capsule Network implementation for CMPT898 - summer 2019
The code is written based on the CapsNet implementation of Aurélien Geron
https://github.com/ageron

'''

from __future__ import division, print_function, unicode_literals

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import random
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import VGG19
from skimage.util import pad

# restart every graph
tf.reset_default_graph()
learning_rate = 0.000005
# random seed (for same results)
np.random.seed(42)
tf.set_random_seed(42)

def get_max(directory):
    # load custom dataset
    import glob
    h= []
    w= []
    address = glob.glob(directory)
    print(directory)
    print(len(address))
    ii = 0
    jj=0
    for img in address:
        n = cv2.imread(img)

        try:
            n.shape
            h.append(n.shape[0])
            w.append(n.shape[1])
            
            ii +=1
        except AttributeError:
            
            jj+=1
    print(ii)
    print(jj)
    return max(h),max(w)


def load_pic(directory, shape, hmax , wmax):
    # load custom dataset
    
    import glob
    cv_img = []
    address = glob.glob(directory)
    for img in address:
        n = cv2.imread(img)
        try:
            if hmax - n.shape[0] > 0 and wmax - n.shape[1]>0:
                pad_width_vertical = hmax - n.shape[0]
                pad_width_horizontal = wmax - n.shape[1]
                pad_top = int(np.floor(pad_width_vertical/2))
                pad_bottom = int(np.ceil(pad_width_vertical/2))
                pad_left =  int(np.floor(pad_width_horizontal/2))
                pad_right = int(np.ceil(pad_width_horizontal/2))
            
                n = pad(n, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), 'constant')
                
                n = cv2.resize(n, shape)
                n = np.array(n)
                cv_img.append(n)
            elif n.shape[1] > n.shape[0]:
                pad_width_horizontal = n.shape[1] - n.shape[0]
                pad_left =  int(np.floor(pad_width_horizontal/2))
                pad_right = int(np.ceil(pad_width_horizontal/2))
            else: 
                pad_width_vertical = n.shape[0] - n.shape[1]  
                pad_top = int(np.floor(pad_width_vertical/2))
                pad_bottom = int(np.ceil(pad_width_vertical/2))
                
        except AttributeError:
            ii = 0

    imgs = np.array(cv_img) / 255.0
    return imgs

#imgs = load_pic(".\YOLOv3\*\*_*_0*.png",(112,112))

def shuffeldata(x, y):
    a = x
    b = y

    c = np.c_[a.reshape(len(a), -1), b.reshape(len(b), -1)]
    np.random.shuffle(c)
    a2 = c[:, :a.size // len(a)].reshape(a.shape)
    b2 = c[:, a.size // len(a):].reshape(b.shape)
    return [a2, b2]

def load_test(shape, test_direct):
#    hmax,wmax = get_max(test_direct + "/*_*_0*.png")
    hmax = 120
    wmax = 140
    print(hmax)
    print(wmax)
    import glob
    folders = glob.glob(test_direct)
    plants =[]
    for folder in folders:
        direction = folder + "/*_*_0*.png"
        new_plants = load_pic(direction, shape, hmax , wmax)

        y = [0] * len(plants)
        plants.append(new_plants)

    y = np.array(y)
    x = plants
    return x, y



directory = "./Data/exg/*"
x_crop,y_crop = load_test((112,112),directory)

def load_pic2(directory, shape):
    # load custom dataset
    import glob
    cv_img = []
    address = glob.glob(directory)
    size = len(address)
    for img in address:
        n = cv2.imread(img)
        n = cv2.resize(n, shape)
        n = np.array(n)
        cv_img.append(n)
    imgs = np.array(cv_img) / 255.0
    return imgs

def load_plant(shape, plant_direct):
    direction = plant_direct + str(1) + '/*.png'
    plants = load_pic2(direction, shape)
    y = [0] * len(plants)
    for ii in range(14):
        if ii < 8:
            direction = plant_direct + str(ii + 2) + '/*.png'
            new_plant = load_pic2(direction, shape)
            

            print(len(new_plant))
            print(direction)
            if len(new_plant)>0:
                y = y + [ii + 1] * len(new_plant)
                plants = np.concatenate((plants, new_plant), axis=0)
        else:
            direction = plant_direct + str(ii + 2) + '/*.png'
            new_plant = load_pic2(direction, shape)
            
            if len(new_plant)>0:
                plants = np.concatenate((plants, new_plant), axis=0)
                y = y + [ii + 1] * len(new_plant)
            print(len(new_plant))
            print(direction)
    y = np.array(y)
    x = plants
    return shuffeldata(x, y)


x_train, y_train = load_plant(shape=(112, 112), plant_direct='./combined_train_rescaled/')

x_test, y_test = load_plant(shape=(112, 112), plant_direct='./combined_val_rescaled/')

x_test = x_crop[0]

print(len(x_crop))
for ii in range(1,len(x_crop)-5):
    x_test = np.concatenate((x_test,x_crop[ii]),axis = 0)
    print(ii)


def load_catdog(shape, dog_direct, cat_direct):
    dogs = load_pic(dog_direct, shape)
    cats = load_pic(cat_direct, shape)
    x = np.concatenate((dogs, cats), axis=0)
    # num_dog = int(len(dogs) / 5)
    # num_cat = int(len(cats) / 5)
    y = np.array([0] * len(dogs) + [1] * len(cats))
    # y = np.array([0, 1, 2, 3] * num_dog + [4] * (len(dogs) - 4 * num_dog) + [5, 6, 7, 8] * num_cat+[9] * (len(cats) - 4 * num_cat))
    return shuffeldata(x, y)


# x_train, y_train = load_catdog(shape=(64, 64),
#                                cat_direct='./train/cat.*.jpg',
#                                dog_direct='./train/dog.*.jpg')
# x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=42)


def load_fashionmnist():
    # the data, shuffled and split between train and test sets
    from keras.datasets import fashion_mnist
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.

    return (x_train, y_train), (x_test, y_test)


def load_mnist():
    # the data, shuffled and split between train and test sets
    from keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.

    return (x_train, y_train), (x_test, y_test)


def load_cifar10():
    # the data, shuffled and split between train and test sets
    from keras.datasets import cifar10
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.reshape(-1, 32, 32, 3).astype('float32') / 255.
    x_test = x_test.reshape(-1, 32, 32, 3).astype('float32') / 255.
    #    x_train = (x_train[:,:,:,1] + x_train[:,:,:,2] + x_train[:,:,:,0])/3
    #    x_test = (x_test[:,:,:,1] + x_test[:,:,:,2] + x_test[:,:,:,0])/3
    return (x_train, y_train), (x_test, y_test)


# (x_train, y_train), (x_test, y_test) = load_cifar10()
# y_train = y_train //5
# y_test = y_test // 5

def batchload(batch_size, x_train, y_train):
    randindex = np.random.choice(x_train.shape[0], batch_size)
    X_batch = []
    Y_batch = []
    for index in randindex:
        xtrind = x_train[index].reshape((-1))
        ytrind = y_train[index].reshape((-1))
        X_batch = np.append(X_batch, xtrind)
        Y_batch = np.append(Y_batch, ytrind)
    X_batch = X_batch.reshape((batch_size, -1))
    Y_batch = Y_batch.reshape((batch_size))
    return X_batch, Y_batch


def batchloadtest(batch_size, x_test, y_test):
    randindex = np.random.choice(x_test.shape[0], batch_size)
    X_batch = []
    Y_batch = []
    for index in randindex:
        xtrind = x_test[index].reshape((-1))
        ytrind = y_test[index].reshape((-1))
        X_batch = np.append(X_batch, xtrind)
        Y_batch = np.append(Y_batch, ytrind)
    X_batch = X_batch.reshape((batch_size, -1))
    Y_batch = Y_batch.reshape((batch_size))
    return X_batch, Y_batch

def batchloadtest2(batch_size, x_test):
    randindex = np.random.choice(x_test.shape[0], batch_size)
    X_batch = []
    for index in randindex:
        xtrind = x_test[index].reshape((-1))
        X_batch = np.append(X_batch, xtrind)
    X_batch = X_batch.reshape((batch_size, -1))
    return X_batch

# load mnist
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/")

##take a look at some MNIST samples
# n_samples = 5
#
# plt.figure(figsize=(n_samples * 2, 3))
# for index in range(n_samples):
#    plt.subplot(1, n_samples, index + 1)
#    sample_image = mnist.train.images[index].reshape(28, 28)
#    plt.imshow(sample_image, cmap="binary")
#    plt.axis("off")
#
# plt.show()
#
# mnist.train.labels[:n_samples]


# Let's builde a CapsNet
# The input placeholder  (28×28 pixels, 1 color channel = grayscale).
imgh = x_train[0].shape[0]
imgw = x_train[0].shape[1]
imgch = x_train[0].shape[2]

X = tf.placeholder(shape=[None, imgh, imgw, imgch], dtype=tf.float32, name="X")

n_epochs = 500
batch_size = 50

caps1_n_caps = 100 
caps1_n_dims = 16
class_num = 15
# First -> Convoloutional layers
# Pre trained layers

# 28x28 -> (after one conve)20x20 -> (after two conves)12x12 -> Strides(2) ->6x6
conv1_params = {
    "filters": 256,
    "kernel_size": 9,
    "strides": 1,
    "padding": "valid",
    "activation": tf.nn.relu,
    "kernel_initializer": 'glorot_uniform'

}

conv2_params = {
    "filters": 256,  # 256 convolutional filters
    "kernel_size": 9,
    "strides": 2,
    "padding": "valid",
    "activation": tf.nn.relu,
    "kernel_initializer": 'glorot_uniform'
}

conv3_params = {
    "filters": 256,  # 256 convolutional filters
    "kernel_size": 5,
    "strides": 2,
    "padding": "valid",
    "activation": tf.nn.relu,
    "kernel_initializer": 'glorot_uniform'
}

conv1 = tf.layers.conv2d(X, name="conv1", **conv1_params)
conv2 = tf.layers.conv2d(conv1, name="conv2", **conv2_params)
conv3 = tf.layers.conv2d(conv2, name="conv3", **conv3_params)

flatten_cnn = tf.layers.flatten(conv3, name="flatten_CNN")

dense_1 = tf.layers.dense(flatten_cnn, 2*caps1_n_caps * caps1_n_dims, activation=tf.nn.sigmoid)
dense_2 = tf.layers.dense(dense_1, caps1_n_caps * caps1_n_dims, activation=tf.nn.sigmoid)
# flaten the output to 6*6*32 (6x6 pic and 32 caps1_n_maps) What???
caps1_raw = tf.reshape(dense_2, [-1, caps1_n_caps, caps1_n_dims],
                       name="caps1_raw")


# Let's do squash! :D
# the length of the vector should be in the [0,1] range

def squash(s, axis=-1, epsilon=1e-7, name=None):
    with tf.name_scope(name, default_name="squash"):
        squared_norm = tf.reduce_sum(tf.square(s), axis=axis,
                                     keep_dims=True)
        safe_norm = tf.sqrt(squared_norm + epsilon)
        squash_factor = squared_norm / (1. + squared_norm)
        unit_vector = s / safe_norm
        return squash_factor * unit_vector


# the primary capsule output
caps1_output = squash(caps1_raw, name="caps1_output")
cpo = caps1_output
# Now let's move on to the Digit capsule
# number of the classes -> 10
# the dimention is 16D
caps2_n_caps = class_num
caps2_n_dims = 16

# Computing the output of the Digit Caps layer by a massive matrix multipication
# creat random weights around zero
init_sigma = 0.1

W_init = tf.random_normal(
    shape=(1, caps1_n_caps, caps2_n_caps, caps2_n_dims, caps1_n_dims),
    stddev=init_sigma, dtype=tf.float32, name="W_init")
W = tf.Variable(W_init, name="W")

# batch_size = tf.shape(X)[0]
W_tiled = tf.tile(W, [batch_size, 1, 1, 1, 1], name="W_tiled")

# the first caps output
caps1_output_expanded = tf.expand_dims(caps1_output, -1,
                                       name="caps1_output_expanded")
caps1_output_tile = tf.expand_dims(caps1_output_expanded, 2,
                                   name="caps1_output_tile")
caps1_output_tiled = tf.tile(caps1_output_tile, [1, 1, caps2_n_caps, 1, 1],
                             name="caps1_output_tiled")

# multiply first caps output to the weights to get prediction results
caps2_predicted = tf.matmul(W_tiled, caps1_output_tiled,
                            name="caps2_predicted")

# the shape of caps2_predicted is (?, 1152, 10, 16, 1) which is a 16 dimentional prediction of 32 dimentional capsules of 6*6 images


# Lets do routing by agreement
# set rowting weights to zero


raw_weights = tf.zeros([batch_size, caps1_n_caps, caps2_n_caps, 1, 1],
                       dtype=np.float32, name="raw_weights")
# raw_weights = tf.random_normal([batch_size, caps1_n_caps, caps2_n_caps, 1, 1],
#                                stddev= init_sigma)
# raw_weights = tf.random.normal([batch_size, caps1_n_caps, caps2_n_caps, 1, 1],
#                        dtype=np.float32, name="raw_weights")

# Round 1 of routing
# calculate weights by applying softmax
routing_weights = tf.nn.softmax(raw_weights, dim=2, name="routing_weights")

# apply weights to prediction results
weighted_predictions = tf.multiply(routing_weights, caps2_predicted,
                                   name="weighted_predictions")
weighted_sum = tf.reduce_sum(weighted_predictions, axis=1, keep_dims=True,
                             name="weighted_sum")

# Squash it!
caps2_output_round_1 = squash(weighted_sum, axis=-2,
                              name="caps2_output_round_1")
# early_pred = weighted_sum
# Round2
# calculate the distance of the prediction to the output vectors
caps2_output_round_1_tiled = tf.tile(
     caps2_output_round_1, [1, caps1_n_caps, 1, 1, 1],
     name="caps2_output_round_1_tiled")

agreement = tf.matmul(caps2_predicted, caps2_output_round_1_tiled,
                       transpose_a=True, name="agreement")

raw_weights_round_2 = tf.add(raw_weights, agreement,
                              name="raw_weights_round_2")

routing_weights_round_2 = tf.nn.softmax(raw_weights_round_2,
                                         dim=2,
                                         name="routing_weights_round_2")
weighted_predictions_round_2 = tf.multiply(routing_weights_round_2,
                                            caps2_predicted,
                                            name="weighted_predictions_round_2")
weighted_sum_round_2 = tf.reduce_sum(weighted_predictions_round_2,
                                      axis=1, keep_dims=True,
                                      name="weighted_sum_round_2")
caps2_output_round_2 = squash(weighted_sum_round_2,
                               axis=-2,
                               name="caps2_output_round_2")
#
# # We should probably add some more rounds
#
#
# # Round3
# # calculate the distance of the prediction to the output vectors
# caps2_output_round_2_tiled = tf.tile(
#     caps2_output_round_2, [1, caps1_n_caps, 1, 1, 1],
#     name="caps2_output_round_2_tiled")
#
# agreement = tf.matmul(caps2_predicted, caps2_output_round_2_tiled,
#                       transpose_a=True, name="agreement")
#
# raw_weights_round_3 = tf.add(raw_weights_round_2, agreement,
#                              name="raw_weights_round_3")
#
# routing_weights_round_3 = tf.nn.softmax(raw_weights_round_3,
#                                         dim=2,
#                                         name="routing_weights_round_3")
# weighted_predictions_round_3 = tf.multiply(routing_weights_round_3,
#                                            caps2_predicted,
#                                            name="weighted_predictions_round_3")
# weighted_sum_round_3 = tf.reduce_sum(weighted_predictions_round_3,
#                                      axis=1, keep_dims=True,
#                                      name="weighted_sum_round_3")
# caps2_output_round_3 = squash(weighted_sum_round_3,
#                               axis=-2,
#                               name="caps2_output_round_3")
#
# # We should probably add some more rounds
caps2_output = caps2_output_round_2


# class probability
def safe_norm(s, axis=-1, epsilon=1e-7, keep_dims=False, name=None):
    with tf.name_scope(name, default_name="safe_norm"):
        squared_norm = tf.reduce_sum(tf.square(s), axis=axis,
                                     keep_dims=keep_dims)
        return tf.sqrt(squared_norm + epsilon)


y_proba = safe_norm(caps2_output, axis=-2, name="y_proba")
early_pred = y_proba
# find the class with highest probability
y_proba_argmax = tf.argmax(y_proba, axis=2, name="y_proba")
y_pred = tf.squeeze(y_proba_argmax, axis=[1, 2], name="y_pred")

# it is done! We have the output of the model

# know the training
# let's keep labels in a placeholder
y = tf.placeholder(shape=[None], dtype=tf.int64, name="y")

# compute the margine loss (complictaed equation ;D)
m_plus = 0.9
m_minus = 0.1
lambda_ = 0.5
# one_hot is creating [0,0,0,1,...,0] for every class (so y should be the number of the class)
T = tf.one_hot(y, depth=caps2_n_caps, name="T")

# comparing the norm of the output with labels
caps2_output_norm = safe_norm(caps2_output, axis=-2, keep_dims=True,
                              name="caps2_output_norm")

# I think the number 10 in shape=(-1, 10) is the number of classes
present_error_raw = tf.square(tf.maximum(0., m_plus - caps2_output_norm),
                              name="present_error_raw")
present_error = tf.reshape(present_error_raw, shape=(-1, class_num),
                           name="present_error")

absent_error_raw = tf.square(tf.maximum(0., caps2_output_norm - m_minus),
                             name="absent_error_raw")
absent_error = tf.reshape(absent_error_raw, shape=(-1, class_num),
                          name="absent_error")

# compute the creepy loss for every instance
L = tf.add(T * present_error, lambda_ * (1.0 - T) * absent_error,
           name="L")

# the whole margine loss
margin_loss = tf.reduce_mean(tf.reduce_sum(L, axis=1), name="margin_loss")

##finally let's reconstruct image to compute second error
# but first, Maksing
mask_with_labels = tf.placeholder_with_default(False, shape=(),
                                               name="mask_with_labels")

reconstruction_targets = tf.cond(mask_with_labels,  # condition
                                 lambda: y,  # if True
                                 lambda: y_pred,  # if False
                                 name="reconstruction_targets")
# creat the mask
reconstruction_mask = tf.one_hot(reconstruction_targets,
                                 depth=caps2_n_caps,
                                 name="reconstruction_mask")

# change the shape of the mask
reconstruction_mask_reshaped = tf.reshape(
    reconstruction_mask, [-1, 1, caps2_n_caps, 1, 1],
    name="reconstruction_mask_reshaped")

# mask the output
caps2_output_masked = tf.multiply(
    caps2_output, reconstruction_mask_reshaped,
    name="caps2_output_masked")

# flaten decoder input

decoder_input = tf.reshape(caps2_output_masked,
                           [-1, caps2_n_caps * caps2_n_dims],
                           name="decoder_input")

# Now let's creat the decoder to reconstruct image based on the masked output

n_hidden1 = 1024
n_hidden2 = 2048
n_output = imgh * imgw * imgch

with tf.name_scope("decoder"):
    hidden1 = tf.layers.dense(decoder_input, n_hidden1,
                              activation=tf.nn.relu,
                              name="hidden1")
    hidden2 = tf.layers.dense(hidden1, n_hidden2,
                              activation=tf.nn.relu,
                              name="hidden2")
    decoder_output = tf.layers.dense(hidden2, n_output,
                                     activation=tf.nn.sigmoid,
                                     name="decoder_output")

# Reconstruction loss

X_flat = tf.reshape(X, [-1, n_output], name="X_flat")
squared_difference = tf.square(X_flat - decoder_output,
                               name="squared_difference")
reconstruction_loss = tf.reduce_mean(squared_difference,
                                     name="reconstruction_loss")

# Final loss

alpha = 0.0005
beta = 0.5
mae = tf.reduce_mean(tf.to_float(tf.math.abs(tf.math.subtract(y,y_pred))))
loss = tf.add(tf.add(margin_loss,beta*mae), alpha * reconstruction_loss, name="loss")

# evaluate models accuracy by conting true classes.

correct = tf.equal(y, y_pred, name="correct")
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")

# now let's creat the optimizer ADAM

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss, name="training_op")

init = tf.global_variables_initializer()
saver = tf.train.Saver()

##Training


restore_checkpoint = False

# n_iterations_per_epoch = int(mnist.train.num_examples) // batch_size
n_iterations_per_epoch = len(x_train) // batch_size

# n_iterations_validation = int(mnist.validation.num_examples) // batch_size
n_iterations_validation = len(x_test) // batch_size
best_loss_val = np.infty
best_acc_val = 0
checkpoint_path = "./my_capsule_network"
total_parameters = 0
for variable in tf.trainable_variables():
    # shape is an array of tf.Dimension
    shape = variable.get_shape()
    # print(shape)
    # print(len(shape))
    variable_parameters = 1
    for dim in shape:
        # print(dim)
        variable_parameters *= dim.value
    # print(variable_parameters)
    total_parameters += variable_parameters
print("This is the number of variables************ : %f" % total_parameters)

'''
with tf.Session() as sess:
    if restore_checkpoint and tf.train.checkpoint_exists(checkpoint_path):
        saver.restore(sess, checkpoint_path)
    else:
        init.run()

    for epoch in range(n_epochs):
        for iteration in range(1, n_iterations_per_epoch + 1):
            X_batch, y_batch = batchload(batch_size, x_train, y_train)
            # Run the training operation and measure the loss:
            _, loss_train = sess.run(
                [training_op, loss],
                feed_dict={X: X_batch.reshape([-1, imgh, imgw, imgch]),
                           y: y_batch,
                           mask_with_labels: True})
            print("\rIteration: {}/{} ({:.1f}%)  Loss: {:.5f}".format(
                iteration, n_iterations_per_epoch,
                iteration * 100 / n_iterations_per_epoch,
                loss_train),
                end="")
        
            # print(y_out)
            # print(y_real)
        # print(qwe)
        # At the end of each epoch,
        # measure the validation loss and accuracy:
        loss_vals = []
        acc_vals = []
        mae_vals = []
        for iteration in range(1, n_iterations_validation + 1):
            X_batch, y_batch = batchloadtest(batch_size, x_test, y_test)
            loss_val, acc_val, mae_val,cor, out = sess.run(
                [loss, accuracy, mae, y, y_pred],
                feed_dict={X: X_batch.reshape([-1, imgh, imgw, imgch]),
                           y: y_batch})
            loss_vals.append(loss_val)
            acc_vals.append(acc_val)
            avgmae = np.mean(mae_val + 0.0)
            mae_vals.append(avgmae)            
            print("\rEvaluating the model: {}/{} ({:.1f}%)".format(
                iteration, n_iterations_validation,
                iteration * 100 / n_iterations_validation),
                end=" " * 10)
        loss_val = np.mean(loss_vals)
        acc_val = np.mean(acc_vals)
        mae_val = np.mean(mae_vals)
        print("\rEpoch: {}  Val accuracy: {:.4f}%  Loss: {:.6f}{}".format(
            epoch + 1, acc_val * 100, loss_val,
            " (improved)" if acc_val > best_acc_val else ""))
        print(mae_val)
        # And save the model if it improved:
        if acc_val > best_acc_val and epoch > 5:
            save_path = saver.save(sess, checkpoint_path)
            best_acc_val = acc_val
            X_batch, y_batch = batchloadtest(batch_size, x_test, y_test)
            cor, out = sess.run(
                [y, y_pred],
                feed_dict={X: X_batch.reshape([-1, imgh, imgw, imgch]),
                           y: y_batch})
            print(cor)
            print(out)
# Evaluation
'''
n_iterations_test =len(x_test) // batch_size

with tf.Session() as sess:
    saver.restore(sess, checkpoint_path)

    out_vals =[]
    out_p = np.zeros((1,50), dtype=int)
    for iteration in range(1, n_iterations_test + 1):
        if (iteration)*batch_size < len(x_test):
            X_batch = x_test[(iteration-1)*batch_size:(iteration)*batch_size]
            out = sess.run(
                [y_pred],
                feed_dict={X: X_batch.reshape([-1, imgh, imgw, imgch])})

            out_vals.append(out)
            out_p = np.concatenate((out_p,out),axis = 1)
            
            print("\rEvaluating the model: {}/{} ({:.1f}%)".format(
                iteration, n_iterations_test,
                iteration * 100 / n_iterations_test),
                end=" " * 10)
    print(out_vals)
    print(len(out_vals))
    print(out_p)
    print(len(out_p))
    np.savetxt("exg.csv", np.transpose(out_p), delimiter=",")




'''
##Prediction
n_samples = 5

sample_images = x_test[0:n_samples]

with tf.Session() as sess:
    saver.restore(sess, checkpoint_path)
    caps2_output_value, decoder_output_value, y_pred_value = sess.run(
            [caps2_output, decoder_output, y_pred],
            feed_dict={X: sample_images,
                       y: np.array([], dtype=np.int64)})
#
#
#Check the dimention effects


def tweak_pose_parameters(output_vectors, min=-0.5, max=0.5, n_steps=11):
    steps = np.linspace(min, max, n_steps) # -0.25, -0.15, ..., +0.25
    pose_parameters = np.arange(caps2_n_dims) # 0, 1, ..., 15
    tweaks = np.zeros([caps2_n_dims, n_steps, 1, 1, 1, caps2_n_dims, 1])
    tweaks[pose_parameters, :, 0, 0, 0, pose_parameters, 0] = steps
    output_vectors_expanded = output_vectors[np.newaxis, np.newaxis]
    return tweaks + output_vectors_expanded

n_steps = 11

tweaked_vectors = tweak_pose_parameters(caps2_output_value, n_steps=n_steps)
tweaked_vectors_reshaped = tweaked_vectors.reshape(
    [-1, 1, caps2_n_caps, caps2_n_dims, 1])


tweak_labels = np.tile(mnist.test.labels[:n_samples], caps2_n_dims * n_steps)

with tf.Session() as sess:
    decoder_output_value = sess.run(
            decoder_output,
            feed_dict={caps2_output: tweaked_vectors_reshaped,
                       mask_with_labels: True,
                       y: tweak_labels})


tweak_reconstructions = decoder_output_value.reshape(
        [caps2_n_dims, n_steps, n_samples, 112, 112])


for dim in range(3):
    print("Tweaking output dimension #{}".format(dim))
    plt.figure(figsize=(n_steps / 1.2, n_samples / 1.5))
    for row in range(n_samples):
        for col in range(n_steps):
            plt.subplot(n_samples, n_steps, row * n_steps + col + 1)
            plt.imshow(tweak_reconstructions[dim, col, row], cmap="binary")
            plt.axis("off")
    plt.show()
'''
##Thats it :D
##Thank you French guy?
#
#
#
