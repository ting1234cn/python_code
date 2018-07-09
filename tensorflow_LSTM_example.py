import tensorflow as tf
from tensorflow.contrib import rnn
from sklearn import preprocessing
import numpy as np
import random
import os
from tensorflow.python import debug as tf_debug
#define constants
#unrolled through 28 time steps
time_steps=30
#hidden LSTM units
num_units=128
n_hidden = 128
#rows of 28 pixels
n_input=13
#learning rate for adam
learning_rate=0.001
#n_class is n dim output.
n_classes=1
#size of batch
batch_size=30
stock_list=["600196","600460","600276","603993"]

#weights and biases of appropriate shape to accomplish above task
out_weights=tf.Variable(tf.random_normal([num_units,n_classes]))
out_bias=tf.Variable(tf.random_normal([n_classes]))
#defining placeholders
#input image placeholder
x=tf.placeholder("float",shape=[None,time_steps,n_input])
#input label placeholder
y=tf.placeholder("float",shape=[None,n_classes])


def gen_data(filename):
    train_x, train_y = [], []
    dataset=np.genfromtxt(filename, dtype=float,usecols= range(1,n_input+2), skip_header=1,autostrip=True)
    x_seq=preprocessing.scale(dataset[:,1:n_input+1])
    #print("scaled x_seq",x_seq)
    y_hat=dataset[:,0]
    #print("y_hat",y_hat)
    # tensorflow的输入必须array必须是n*n（2*2或者3*3），即行和列数必须一致
    for i in range(len(x_seq)-time_steps):
        feature=np.asarray([x_seq[i+j] for j in range(time_steps)])
        train_x.append(feature)

    train_y=y_hat[time_steps:len(x_seq),np.newaxis].tolist()
 #   train_y.append(label)
    return train_x, train_y



#processing the input tensor from [batch_size,n_steps,n_input] to "time_steps" number of [batch_size,n_input] tensors
#input=tf.expand_dims(x,0)
input=tf.unstack(x,axis=1)


#defining the network
#这里原文写的是n_hidden，其实不应当！应该把n_hidden改为num_units
#或者在前面再定义一个n_hidden = 128

lstm_layer=rnn.BasicLSTMCell(n_hidden,forget_bias=1)
outputs,_=rnn.static_rnn(lstm_layer,input,dtype="float32")

#converting last output of dimension [batch_size,num_units] to [batch_size,n_classes] by out_weight multiplication
prediction=tf.matmul(outputs[-1],out_weights)+out_bias

#loss_function
#loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction,labels=y))
loss=tf.reduce_mean(tf.square(prediction-y))
#optimization
opt=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

#initialize variables
init=tf.global_variables_initializer()
saver=tf.train.Saver()

def train(stock,new_model=False):

    feature, label = gen_data(stock+".txt")
    with tf.Session() as sess:
    #  sess = tf_debug.LocalCLIDebugWrapperSession(sess=sess)
        sess.run(init)
        iter=1
        steps=1
        if new_model==False and os.path.exists("./"+stock+"/"+stock+"model.ckpt.meta"):
            saver.restore(sess,"./"+stock+"/"+stock+"model.ckpt")


        while iter <= len(feature) - time_steps:

            batch_x = feature[iter:iter + batch_size]
            batch_y = label[iter:iter + batch_size]
            sess.run(opt, feed_dict={x: batch_x, y: batch_y})

            if steps % 1000 == 0:
                # sess.run(tf.Print(y,[y],"y value is"))
                los = sess.run(loss, feed_dict={x: batch_x, y: batch_y})
                saver.save(sess, "./"+stock+"/"+stock+"model.ckpt")

                print(stock+" for step ", steps)
                print(stock+" Loss ", los)
                print("__________________")
                if los < 0.001:
                    break
            iter = random.randint(0, len(feature) - time_steps)
            steps = steps + 1




if __name__ == '__main__':
    writer = tf.summary.FileWriter(r'C:\Users\twan\tf', tf.get_default_graph())
    for stock in stock_list:
        train(stock,new_model=False)
    writer.close()