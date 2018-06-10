import numpy as np
import tensorflow as tf

tf.set_random_seed(777)

xy = np.loadtxt("텍스트파일이름.csv",delimiter = ',',
                dtype = np.float32)
x_data = xy[:,0:-1]
y_data = xy[:,[-1]]

print(x_data.shape,x_data,len(x_data))
print(y_data.shape,y_data)

X = tf.placeholder(tf.float32,shape = [None,3])
Y = tf.placeholder(tf.float32,shape = [None,1])

w = tf.Variable(tf.random_normal([3,1]),name = 'weight')
b = tf.Variable(tf.random_normal([1]),name'bias')

hypothesis = tf.matmul(w,X) + b

cost = tf.reduce_mean(tf.square(hypothesis - Y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01)
train = optimizer.minimize(cost)

sess = tf.Session()

sess.run(tf.global_variables_initializer())

for step in range(2001):
    cost_val, hy_val, _ = sess.run([cost,hypothesis,train]
                                   feeddict = {X: x_data,Y = y_data})
    if step % 10 == 0 ;
    print(step,"cost: ",cost_val ,
          "\n prediction : \n", hy_val)

