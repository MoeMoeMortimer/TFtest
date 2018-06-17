#coding=utf-8
import tensorflow as tf
#import input_data # 调用input_data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('data/', one_hot=True)
#设定参数
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

#建立模型
y = tf.nn.softmax(tf.matmul(x,W) + b)

#实际值
y_ = tf.placeholder("float", [None,10])

#定义cost
cross_entropy = -tf.reduce_sum(y_*tf.log(y))

#设定训练算法
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

#初始化变量
init = tf.global_variables_initializer()

#启动模型
sess = tf.Session()
sess.run(init)

#施行训练
for i in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

#建立评估模型
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

#布尔值转换成浮点数，然后取平均值
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

#计算所学习到的模型在测试数据集上面的正确率
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
