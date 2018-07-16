from tensorflow_LSTM_example import *

def predict_all():
    writer = tf.summary.FileWriter(r'C:\Users\twan\tf', tf.get_default_graph())
    for stock in stock_list:
        predict(stock)
    writer.close()

def predict(stock):
    feature, label = gen_data(stock + ".txt")
    with tf.Session() as sess:
        #  sess = tf_debug.LocalCLIDebugWrapperSession(sess=sess)
        sess.run(init)
        if os.path.exists("./" + stock + "/" + stock + "model.ckpt.meta"):
            saver.restore(sess, "./" + stock + "/" + stock + "model.ckpt")
        else:
            print(stock + " model data does not exist")
            return
        start = len(feature) - time_steps
        end = len(feature)
        predict_value = sess.run(prediction, feed_dict={x: feature[start:end]})
        print(stock + " prediction", predict_value[-1])
        # print("target",label[start:end])


if __name__ == '__main__':
    predict_all()
    #predict("002415")