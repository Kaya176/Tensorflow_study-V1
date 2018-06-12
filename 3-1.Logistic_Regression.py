import tensorflow as tf
import numpy as np

xy = np.loadtxt("titanic_full.csv",delimiter = ",",dtype = np.float32)
x_data = xy[:,1:4]
y_data = xy[:,[0]]

#우리가 넣을 데이터 x <- 독립변수  / y<-종속변수
#x_data = [[1,2],[2,3],[3,1],[4,3],[5,3],[6,2]]
#y_data = [[0],[0],[0],[1],[1],[1]]
#feeddict를 사용하기 위해 미리 설정해둠.
#x데이터의 형태에 주의하여 [None,size of x] 설정
#[몇개의 데이터가 들어가나? ->None : 모름 /
#들어가는 데이터의 크기는? ]
X = tf.placeholder(tf.float32, shape = [None,3])
Y = tf.placeholder(tf.float32, shape = [None,1])
#우리가 예측하고자 하는 값
# tf.random_normal([들어오는 데이터 크기, 나가는  데이터 크기])
w = tf.Variable(tf.random_normal([3,1]), name = "weight")

b = tf.Variable(tf.random_normal([1]),name = "bias")

#행렬의 곱은 크기에 주의해야함
hypothesis = tf.sigmoid(tf.matmul(X,w) + b)

#Cost Function
cost = -tf.reduce_mean(Y*tf.log(hypothesis) + (1-Y)*tf.log(1-hypothesis))

train = tf.train.GradientDescentOptimizer(learning_rate = 0.01).minimize(cost)
#optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01)
#train = optimizer.minimize(cost)
#라고 써도 됨.

#Accuracy Computation
#True if hypothesis > 0.5 else false
predicted = tf.cast(hypothesis >0.5,dtype = tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted,Y),dtype = tf.float32))

#Launch graph
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(10001):
        cost_val,_ = sess.run([cost,train],feed_dict={X : x_data,Y:y_data})
        if step % 200 == 0:
            print(step,cost_val)
    h,c,a = sess.run([hypothesis,predicted,accuracy],
                     feed_dict = {X:x_data,Y:y_data})
    print("\n Hypothesis: ",h,"\nCorrect(Y) : ",c,"\nAccuracy: ",a)
    
    
