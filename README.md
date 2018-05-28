# Learning_TensorFlow
이 저장소는 텐서플로우에 대해 공부하고 코드를 공유 및 데이터자료를 공유하는 저장소 입니다.

정확히 쓰려고 노력은 하였지만 100% 정확한 내용이 있다고 확신할 순 없습니다.

목적은 개인의 학습을 위해 만들었지만 다른 분들도 마음껏 공유 가능하십니다.


목차

1. linear Regression

2. Logistic (Regression) Classification

3. Softmax Regression (Multinomial Logistic Regression)

4. Basic of Deep learning

5. Neural Network 1: XOR 문제와 학습방법, Backpropagation

6. Convolutional Neural Networks

7. Recurrent Neural Network

-----------------------------------------------------------------

1. Linear Regression

선형 회귀 분석이란 종속변수 Y에 대해 X라는 독립변수가 있어서 Y= a*X +b 꼴의 선형 상관관계를 모델링 하는 분석 기법이다.

우선 Tensorflow를 improt 시켜놓고, 간단한 모델 X= [1,2,3]과 Y = [1,2,3]을 사용하여 간단한 선형회귀선을 텐서플로우를 이용하여 만들어 보자.

첫번째로 Y = W*X + b라는 선형 회귀선이 있다고 한다면, 우리는 적절한 W값과 b값을 찾기를 원한다.

따라서 다음과 같이 X~N(0,1)을 따르는 정규분포 난수를 이용하여 랜던한 값 하나를 설정하고자 한다.

코드는 다음과 같다.

W = tf.Variable(tf.random_normal([1]),name = "weight")

b = tf.Variable(tf.random_normal([1]),name = "bias")

위 코드는 W와 b값을 임의로 하나 설정하는 코드이고 Variable이라는 것은 텐서플로우에서 어려차례 트레이닝을 할때마다 값이 변하는 텐서변수라고 이해하면 된다.

다음으로 우리가 처음에 설정한 가설을 입력한다.

hypothesis = x_train*W + b

다음으로 Cost function을 이용해야한다.



-Cost function이란? 

우리는 바로 위에 가설 H(x) = W*x_train + b을 세웠다. 하지만 이는 어디까지나 가설에 불과하다.

실제값이 Y(1)이라고 한다면 우리는 여기에 우리가 세운 가설에 X(1)을 대입한 H(1)을 구할 수 있는데, 이떄, 실제 값과 예측값과의 차이(Error)
가 발생하게 된다. 이때 최소제곱법을 참고히면, 

Error^2 = (Y(1) - H(1))^2

이 된다. 이때, Cost 함수라는것은 이러한 Error의 제곱의 합을 구하고 이를 노드의 갯수(m)로 나눈것으로 정의한다.



따라서 위와 같은 Cost함수를 이용하여 코드를 짜게 된다면 다음과 같다.

cost = tf.reduce_mean(tf.square(hypothesis - y_train))

위에서 tf.reduce_mean은 수식적으로 Sigma와 1/m을 나타낸다.

다음으로 위에서 구한 cost값이 최소가 되는 W와 b값을 찾아야 하는데 이는 '경사하강법'을 이용한다.

optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01)

train = optimizer.minimize(cost)

위의 코드는 나중에 따로 자세히 언급하도록 한다.

다음으로 세션을 정의하고 training시키면 완성된다.

sess = tf.Session()

sess.run(tf.global_variables_initilizer())

for step in range(2001):
    sess.run(train)
    if step % 20 == 0:
        print(step,sess.run(cost),sess.run(W),sess.run(b))

총 2000번을 반복하는데, 이를 화면에 전부 출력하기엔 양이 너무 많으므로 20씩 건너 뛰면서 출력을 하도록 한다.

