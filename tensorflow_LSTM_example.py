import tensorflow as tf
from tensorflow.contrib import rnn
from sklearn import preprocessing
import numpy as np
#import mnist dataset
from tensorflow.examples.tutorials.mnist import input_data
#mnist=input_data.read_data_sets("/tmp/data/",one_hot=True)



#define constants
#unrolled through 28 time steps
time_steps=1
#hidden LSTM units
num_units=128
n_hidden = 128
#rows of 28 pixels
n_input=14
#learning rate for adam
learning_rate=0.001
#mnist is meant to be classified in 10 classes(0-9).
n_classes=1
#size of batch
batch_size=11
#weights and biases of appropriate shape to accomplish above task
out_weights=tf.Variable(tf.random_normal([num_units,n_classes]))
out_bias=tf.Variable(tf.random_normal([n_classes]))
#defining placeholders
#input image placeholder
x=tf.placeholder("float",[None,time_steps,n_input])
#input label placeholder
y=tf.placeholder("float",[None,n_classes])

def gen_data(filename):
    dataset=np.genfromtxt(filename, dtype=float,usecols= range(1,n_input+2), skip_header=1,autostrip=True)
    x_seq=preprocessing.scale(dataset[:,1:n_input+1])
    print("scaled x_seq",x_seq)
    y_hat=dataset[:,0]
    print("y_hat",y_hat)
 #   print("Y_hat",y_hat)
    return x_seq, y_hat

x_seq, y_hat = gen_data("600196.txt")

#processing the input tensor from [batch_size,n_steps,n_input] to "time_steps" number of [batch_size,n_input] tensors
input=tf.unstack(x,time_steps,1)

#defining the network
#这里原文写的是n_hidden，其实不应当！应该把n_hidden改为num_units
#或者在前面再定义一个n_hidden = 128

lstm_layer=rnn.BasicLSTMCell(n_hidden,forget_bias=1)
outputs,_=rnn.static_rnn(lstm_layer,input,dtype="float32")

#converting last output of dimension [batch_size,num_units] to [batch_size,n_classes] by out_weight multiplication
prediction=tf.matmul(outputs[-1],out_weights)+out_bias

#loss_function
loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))
#optimization
opt=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
#model evaluation
correct_prediction=tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

#initialize variables
init=tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    iter=1
    while iter < 10:
        batch_x = x_seq[(iter-1)*batch_size:iter*batch_size]
        batch_y = y_hat[(iter-1)*batch_size:iter*batch_size]

        batch_x = batch_x.reshape((batch_size, time_steps, n_input))
        sess.run(opt, feed_dict={x: batch_x, y: batch_y})
        if iter % 10 == 0:
            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
            los = sess.run(loss, feed_dict={x: batch_x, y: batch_y})
            print("For iter ", iter)
            print("Accuracy ", acc)
            print("Loss ", los)
            print("__________________")
        iter = iter + 1
        #记得这一段要缩进到session里面
        #calculating test accuracy
     #   test_data = mnist.test.images[:128].reshape((-1, time_steps, n_input))
     #  test_label = mnist.test.labels[:128]
     # print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: test_data, y: test_label}))

    writer = tf.summary.FileWriter(r'C:\Users\twan\tf', tf.get_default_graph())
    writer.close()