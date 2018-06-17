#coding=utf-8
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
#import input_data # 调用input_data
mnist = input_data.read_data_sets('data/', one_hot=True)

#权重初始化
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

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
  
#卷积和池化  W：卷积盒大小，步长strides=[a,b,c,d]  b:往左移个数，c:往右移  SAME：表原来图片的大小除以步长
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

#这个盒的大小是2*2的
def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
						
#第一层卷积  5*5卷积盒大小，1：通道数（背白的），
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])  #权重
x_image = tf.reshape(x, [-1,28,28,1])  #
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)  #激活函数，正向抑制的
h_pool1 = max_pool_2x2(h_conv1)
#第一层变成14

#第二层卷积   通道数变了
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

#密集连接层  1024个神经元
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

#Dropout
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#输出层
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


#训练和评估模型

cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))



#启动模型
sess = tf.Session()
sess.run(tf.global_variables_initializer())
for i in range(1000):
    batch = mnist.train.next_batch(50)
    sess.run(train_step, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
