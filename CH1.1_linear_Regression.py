import tensorflow as tf

def hello():
    node1 = tf.constant(3.0, tf.float32)
    node2 = tf.constant(4.0) #alse tf.float32 implicitly
    node3 = tf.add(node1,node2)

    print("node1 :",node1)
    print("node2 :",node2)
    print("node3 :", node3)

    sess = tf.Session()
    print("sess.run(node1,node2) : ",sess.run([node1,node2]))
    print("sess,run(node3) : ",sess.run(node3))


def placeholder():

    a = tf.placeholder(tf.float32)
    b = tf.placeholder(tf.float32)

    adder_node = a + b #위에서 정의한 두 텐서를 더하는 노드를 만듬
    sess = tf.Session()
    print(sess.run(adder_node, feed_dict = {a: 3, b:4.5}))
    print(sess.run(adder_node, feed_dict = {a: [1,3],b:[4,5]}))


def train1():
    x_train = [1,2,3]
    y_train = [1,2,3]

    W = tf.Variable(tf.random_normal([1]),name = "weight")
    b = tf.Variable(tf.random_normal([1]),name = "bias")
#our hypothesis
    hypothesis = x_train * W + b
#cost
    cost = tf.reduce_mean(tf.square(hypothesis - y_train)) #reduce_mean = 오차의 제곱의 평균을 내줌
#Minimize
    optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01)
    train = optimizer.minimize(cost) 

    sess = tf.Session()
#Initialize global variables in the graph
    sess.run(tf.global_variables_initializer())
#Fit the line
    for step in range(2001):
        sess.run(train)
        if step % 20 == 0:
            print(step,sess.run(cost),sess.run(W),sess.run(b))


#placeholder를 이용하여 트레이닝 해보기
def train2():
    W = tf.Variable(tf.random_normal([1]),name= "Weight")
    b = tf.Variable(tf.random_normal([1]),name= "bias")
    X = tf.placeholder(tf.float32,shape = [None])
    Y = tf.placeholder(tf.float32,shape = [None])

    hypothesis = X * W + b

    cost = tf.reduce_mean(tf.square(hypothesis - Y))

    optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01)
    train = optimizer.minimize(cost) 
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    for step in range(2001):
        cost_val,w_val,b_val, _ = sess.run([cost,W,b,train],
                                           feed_dict = {X:[1,2,3,4,5],Y:[2.1,3.2,4.2,5.2,6.2]})
        #if step % 20 == 0:
           # print(step,cost_val,w_val,b_val)
    print(sess.run(hypothesis,feed_dict={X:[5]}))
    print(sess.run(hypothesis,feed_dict={X:[1.2,3,45]}))
