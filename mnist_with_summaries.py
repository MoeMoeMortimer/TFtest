# Copyright 2015 The TensorFlow Authors. All Rights Reserved.  
#  
# Licensed under the Apache License, Version 2.0 (the 'License');  
# you may not use this file except in compliance with the License.  
# You may obtain a copy of the License at  
#  
#     http://www.apache.org/licenses/LICENSE-2.0  
#  
# Unless required by applicable law or agreed to in writing, software  
# distributed under the License is distributed on an 'AS IS' BASIS,  
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
# See the License for the specific language governing permissions and  
# limitations under the License.  
# ==============================================================================  
"""A simple MNIST classifier which displays summaries in TensorBoard. 
 
 This is an unimpressive MNIST model, but it is a good example of using 
tf.name_scope to make a graph legible in the TensorBoard graph explorer, and of 
naming summary tags so that they are grouped meaningfully in TensorBoard. 
 
It demonstrates the functionality of every TensorBoard dashboard. 
"""  
from __future__ import absolute_import  
from __future__ import division  
from __future__ import print_function  
  
import argparse  
import sys  
  
import tensorflow as tf  
  
from tensorflow.examples.tutorials.mnist import input_data  
  
FLAGS = None  
  
  
def train():  
  # Import data加载数据
  mnist = input_data.read_data_sets(FLAGS.data_dir,  
                                    one_hot=True,  
                                    fake_data=FLAGS.fake_data)  
  
  sess = tf.InteractiveSession()   # 使用在交互式上下文环境的tf会话
  
  # Create a multilayer model.  
  
  # Input placeholders（占位符）
  # 定义两个【占位符】，作为【训练样本图片/此块样本作为特征向量存在】和【类别标签】的输入变量，并将这些占位符存在命名空间input中
  with tf.name_scope('input'):  
    x = tf.placeholder(tf.float32, [None, 784], name='x-input')
    # x不是一个特定的值，而是一个占位符placeholder，我们在TensorFlow运行计算时输入这个值。我们希望能够输入任意数量的MNIST图像，每一张图展平成784维的向量。我们用2维的浮点数张量来表示这些图，这个张量的形状是[None，784 ]。
    y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')
    # 为了计算交叉熵，添加一个新的占位符用于输入正确值
  #将【输入的特征向量】还原成【图片的像素矩阵】，并通过tf.summary.image函数定义将当前图片信息作为写入日志的操作
  with tf.name_scope('input_reshape'):  
    image_shaped_input = tf.reshape(x, [-1, 28, 28, 1]) 
    # -1表示任意数量的样本数,大小为28x28深度为一的张量(其实是用深度为28的,28x1的张量,来表示28x28深度为1的张量)
    tf.summary.image('input', image_shaped_input, 10)
   
  #=======================================================================================================================
  #函数说明：
  #       初始化所有权值
  #=======================================================================================================================
  # We can't initialize these variables to 0 - the network will get stuck.  
  def weight_variable(shape):  
    """Create a weight variable with appropriate initialization."""  
    initial = tf.truncated_normal(shape, stddev=0.1)  
    return tf.Variable(initial)  
  #=======================================================================================================================
  #函数说明：
  #       初始化所有偏置项
  #=======================================================================================================================
  def bias_variable(shape):  
    """Create a bias variable with appropriate initialization."""  
    initial = tf.constant(0.1, shape=shape)  
    return tf.Variable(initial)  
  #=======================================================================================================================
  #函数说明：
  #       生成【变量】的监控信息，并将生成的监控信息写入【日志文件】
  #参数说明：
  #       [1]var :需要【监控】和【记录】运行状态的【张量】
  #       [2]name:给出了可视化结果中显示的图表名称
  #scalar函数原型:
  #       def scalar(name, tensor, collections=None, family=None)
  #=======================================================================================================================
  def variable_summaries(var):  
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""  
    with tf.name_scope('summaries'):  
      mean = tf.reduce_mean(var)
      tf.summary.scalar('mean', mean)  
      with tf.name_scope('stddev'):  
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))  
      tf.summary.scalar('stddev', stddev)  
      tf.summary.scalar('max', tf.reduce_max(var))  
      tf.summary.scalar('min', tf.reduce_min(var))  
      tf.summary.histogram('histogram', var)  
  #=======================================================================================================================
  #函数说明：
  #       生成一层全连接层神经网络
  #=======================================================================================================================
  def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):  
    """Reusable code for making a simple neural net layer. 
    
    It does a matrix multiply, bias add, and then uses relu to nonlinearize. 
    It also sets up name scoping so that the resultant graph is easy to read, 
    and adds a number of summary ops. 
    """  
    # Adding a name scope ensures logical grouping of the layers in the graph.  
    with tf.name_scope(layer_name):  
      # This Variable will hold the state of the weights for the layer  
      with tf.name_scope('weights'):  
        weights = weight_variable([input_dim, output_dim])  
        variable_summaries(weights)  
      with tf.name_scope('biases'):  
        biases = bias_variable([output_dim])  
        variable_summaries(biases)  
      with tf.name_scope('Wx_plus_b'):  
        preactivate = tf.matmul(input_tensor, weights) + biases  
        tf.summary.histogram('pre_activations', preactivate)  
      activations = act(preactivate, name='activation')  
      tf.summary.histogram('activations', activations)  
      return activations  
  
  # 第一层 
  hidden1 = nn_layer(x, 784, 500, 'layer1')  
  
  #dropout
  with tf.name_scope('dropout'):  
    keep_prob = tf.placeholder(tf.float32)  
    tf.summary.scalar('dropout_keep_probability', keep_prob)  
    dropped = tf.nn.dropout(hidden1, keep_prob)  
  
  # 第二层
  # Do not apply softmax activation yet, see below.  
  y = nn_layer(dropped, 500, 10, 'layer2', act=tf.identity)  
  
  with tf.name_scope('cross_entropy'):  
    # The raw formulation of cross-entropy,  
    #  
    # tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.softmax(y)),  
    #                               reduction_indices=[1]))  
    #  
    # can be numerically unstable.  
    #  
    # So here we use tf.nn.softmax_cross_entropy_with_logits on the  
    # raw outputs of the nn_layer above, and then average across  
    # the batch.  
    diff = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)  
    with tf.name_scope('total'):  
      cross_entropy = tf.reduce_mean(diff)  
  tf.summary.scalar('cross_entropy', cross_entropy)  
  
  with tf.name_scope('train'):
    # AdamOptimizer控制学习速度  minimize让损失最小
    train_step = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(  
        cross_entropy)  
  
  with tf.name_scope('accuracy'):  
    with tf.name_scope('correct_prediction'):  
      correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))  
    with tf.name_scope('accuracy'):  
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  
  tf.summary.scalar('accuracy', accuracy)  
  
  # Merge all the summaries and write them out to /tmp/mnist_logs (by default)  
  merged = tf.summary.merge_all()  
  # 实例化FileWriter的类对象，并将当前TensoirFlow的计算图写入【日志文件】
  train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)  
  test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test')  
  # Tensorflow中创建的变量，在使用前必须进行初始化，下面这个为初始化函数
  tf.global_variables_initializer().run()  
  
  # Train the model, and also write summaries.  
  # Every 10th step, measure test-set accuracy, and write test summaries  
  # All other steps, run train_step on training data, & add training summaries  
  
  def feed_dict(train):  
    """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""  
    if train or FLAGS.fake_data:  
      xs, ys = mnist.train.next_batch(100, fake_data=FLAGS.fake_data)  
      k = FLAGS.dropout  
    else:  
      xs, ys = mnist.test.images, mnist.test.labels  
      k = 1.0  
    return {x: xs, y_: ys, keep_prob: k}  
  
  # 10的位数为测试集，其他为训练集
  for i in range(FLAGS.max_steps):  
    if i % 10 == 0:  # Record summaries and test-set accuracy
      # 运行训练步骤以及所有的【日志文件生成操作】，得到这次运行的【日志文件】。
      summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))  
      # 将所有的日志写入文件，TensorFlow程序就可以那这次运行日志文件，进行各种信息的可视化
      test_writer.add_summary(summary, i)  
      print('Accuracy at step %s: %s' % (i, acc))  
    else:  # Record train set summaries, and train  
      if i % 100 == 99:  # Record execution stats    100次记录一次
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)  
        run_metadata = tf.RunMetadata()  
        summary, _ = sess.run([merged, train_step],  
                              feed_dict=feed_dict(True),  
                              options=run_options,  
                              run_metadata=run_metadata)  
        train_writer.add_run_metadata(run_metadata, 'step%03d' % i)  
        # 将所有的日志写入文件，TensorFlow程序就可以那这次运行日志文件，进行各种信息的可视化
        train_writer.add_summary(summary, i)
        print('Adding run metadata for', i)  
      else:  # Record a summary  
        summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True))  
        train_writer.add_summary(summary, i)  
  train_writer.close()  
  test_writer.close()  
  
  
def main(_):  
  if tf.gfile.Exists(FLAGS.log_dir):  
    tf.gfile.DeleteRecursively(FLAGS.log_dir)  
  tf.gfile.MakeDirs(FLAGS.log_dir)  
  train()  
  
  
if __name__ == '__main__':  
  parser = argparse.ArgumentParser() # 设置一个解析器，argparse是对象命令行解析模块  
  # 向该对象中添加命令行参数和选项
  parser.add_argument('--fake_data', nargs='?', const=True, type=bool,  
                      default=False,  
                      help='If true, uses fake data for unit testing.')  
  parser.add_argument('--max_steps', type=int, default=1000,  
                      help='Number of steps to run trainer.')  
  parser.add_argument('--learning_rate', type=float, default=0.001,  
                      help='Initial learning rate')  
  parser.add_argument('--dropout', type=float, default=0.9,  
                      help='Keep probability for training dropout.')  
  parser.add_argument('--data_dir', type=str, default='MNIST_data',  
                      help='Directory for storing input data')  
  parser.add_argument('--log_dir', type=str, default='F:/l',  
                      help='Summaries log directory')  
  FLAGS, unparsed = parser.parse_known_args()  # 进行解析
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)  