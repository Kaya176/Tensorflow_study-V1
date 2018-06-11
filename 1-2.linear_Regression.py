import tensorflow as tf
#Multy Variable linear Regression
x1_data = [73.,93.,89.,96.,73.]
x2_data = [80.,88.,91.,98.,66.]
x3_data = [75.,93.,90.,100.,70.]
y_data = [152.,185.,180.,196.,142.]

#place holder for a tensor that will be always feed.
x1 = tf.placeholder(tf.float32,shape = [None])
x2 = tf.placeholder(tf.float32,shape = [None])
x3 = tf.placeholder(tf.float32,shape = [None])

y = tf.placeholder(tf.float32,shape = [None])

w1 = tf.Variable(tf.random_normal([1]),name = "weight1")
w2 = tf.Variable(tf.random_normal([1]),name = "weight2")
w3 = tf.Variable(tf.random_normal([1]),name = "weight3")

b = tf.Variable(tf.random_normal([1]),name = "bias")
hypothesis = x1*w1 + x2*w2 + x3*w3 + b

#cost/Loss ftn
cost = tf.reduce_mean(tf.square(hypothesis - y))

#minimiz1. Need a very small learning rate for this data set
optimizer = tf.train.GradientDescentOptimizer(learning_rate= 1e-5)
train = optimizer.minimize(cost)

#Lunch the graph in a Session

sess = tf.Session()
#Initialize global variables in the gruop
sess.run(tf.global_variables_initializer())
for step in range(2001):
    cost_val,hy_val, _ = sess.run([cost, hypothesis, train],
                                  feed_dict = {x1: x1_data, x2: x2_data, x3:x3_data, y : y_data})
    if step % 20 == 0 :
        print(step,"cost :",cost_val , "\n prediction \n",hy_val)
