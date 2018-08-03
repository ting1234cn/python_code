from __future__ import unicode_literals
from tensorflow_LSTM_example import *
import matplotlib.pyplot as plt


def predict_all():
    current_date = time.strftime("%F")
    print(current_date)
    writer = tf.summary.FileWriter(r'C:\Users\twan\tf', tf.get_default_graph())
    for stock in stock_list.keys():
        predict(stock)
    writer.close()

def predict(stock,show_pic=True,show_annotation=False):
    feature, label,date_index = load_data(stock + ".txt",preprocess=True)
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
        if stock in stock_list:
            print(stock_list[stock] + " prediction", predict_value[-1])
        else:
            print(stock + " prediction ", predict_value[-1])
        # print("target",label[start:end])
        if show_pic:
            plt.figure(stock)
            label.append([label[end-n_output-1][1],0])
            y=np.asarray(label[start:end-n_output+1])[:,0]

            plt.plot(list(range(len(y))),y,color="g",label="actual",marker=">")
            plt.plot(list(range(time_steps)),predict_value[:,0],color="r",label="predict",marker="o")
            #plt.plot(list(date_index[start:end-n_output+2]), predict_value[:, 0], color="r", label="predict", marker="o")
            plt.legend(loc="upper right")
            plt.ylabel("high price")
            #plt.xticks(list(date_index[start:end-n_output+2]))
            #plt.plot(list(range(time_steps)), predict_value[:,1],label="predict2", color="g")
            plt.title(stock+stock_list[stock],fontproperties="SimHei")
            if show_annotation==True:
                for xy1 in zip(range(time_steps),predict_value[:,0] ):  # 标注数据
                    plt.annotate("%.2f" % xy1[1], xy=xy1, xytext=(-1, 5), textcoords='offset points', color='r')

            plt.text(time_steps,predict_value[-1,0] ,str(predict_value[-1,0]))
            plt.savefig(fname=stock)
            plt.show()



if __name__ == '__main__':
    predict_all()
    #predict("600177")