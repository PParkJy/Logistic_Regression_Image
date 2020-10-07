import os, cv2  # cv2 -- OpenCV
import numpy as np

# import gzip
# import six.moves.cPickle as picklr
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import np_utils
# from sklearn.linear_model import LogisticRegressionCV - sklearn에서 제공하는 라이브러리
import tensorflow as tf

train_path = './train_balance/'
test_path = './test/'
classes = ["bacteria", "non_bacteria"]

num_classes = len(classes)  # 클래스=종속변수=label의 개수 -> 2개이므로 binary classification
width = 32
height = 32
channel = 3

X_train = []  # train 이미지
Y_train = []  # train label
X_test = []  # test 이미지
Y_test = []  # test label


def load_image(paths, x_list, y_list, l):
    for top, dir, f in os.walk(paths):
        for filename in f:
            print(paths + filename)
            img = cv2.imread(paths + filename)
            img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA).flatten()
            x_list.append([np.array(img).astype('float32')])
            y_list.append(l)


for idx, label in enumerate(classes):
    labeling = [0 for i in range(num_classes)]
    labeling[idx] = 1
    train_image = train_path + label + "/"
    test_image = test_path + label + "/"

    load_image(train_image, X_train, Y_train, labeling)  # train 이미지 로드
    load_image(test_image, X_test, Y_test, labeling)  # test 이미지 로드

# 픽셀 1개는 1byte 사용 = 8비트 = 256가지를 표현 가능 -> 256으로 나눠서 0~1 사이의 값으로 표현
# RGB 세가지가 있으므로 256X256X256가지의 색상을 표현 가능
X_train = np.vstack(X_train) / 256.0
Y_train = np.array(Y_train).astype('float32')
X_test = np.vstack(X_test) / 256.0
Y_test = np.array(Y_test).astype('float32')

print(
    X_train.shape)  # (11896,3072) -> 11896 = trainset의 개수 (bacteria : 6808, healthy: 5088), 3072 = 특징의 개수 = 32(=이미지 너비) * 32(=이미지 높이) * 3(=RGB값)
print(Y_train.shape)  # (11896,2) -> 2 = label의 개수(bacteria, healthy)
print(X_test.shape)  # (2972.3072) ->2972 = trainset의 개수
print(Y_test.shape)  # (2972,2)

def logistic_model(input_dim, output_dim):
    model = Sequential()
    model.add(Dense(output_dim, input_dim=input_dim,activation='sigmoid'))
    return model

batch_size = 100
epoch = 10
num_label =2
input_dim = 3072

def image_to_feature_vector(image,size=(32,32)):
    return cv2.resize(image,size).flatten()
model = logistic_model(input_dim,num_label)
model.summary()

model.compile(optimizer='sgd', loss='binary_crossentropy',metrics=['accuracy'])
history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=epoch, verbose=1, validation_data=(X_test,Y_test))
score = model.evaluate(X_test,Y_test,verbose=0)
print('Test score: ', score[0])
print('Test accuracy: ',score[1])

json_string = model.to_json()
open('./weight/bacteria_healthy.json','w').write(json_string)
model.save_weights('./weight/bacteria_healthy.h5')


'''
N = X_train.shape[0]

X = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])  # X = 독립변수
X = tf.reshape(X, [-1, 3072])
Y = tf.placeholder(tf.float32, shape=[None, 2])  # binary classification
W = tf.Variable(tf.zeros([3072, 2]))  # weight
b = tf.Variable(tf.zeros([2]))  # bias

Z = tf.matmul(X, W) + b  # 많이 본 linear regression의 형태
A = tf.sigmoid(Z)  # logistic이므로 sigmoid 형태의 activation 함수 사용 (logistic = linear + sigmoid)
cost = -tf.reduce_mean(Y * tf.log(A) + (1 - Y) * tf.log(1 - A))  # 크로스 엔트로피를 코스트 함수로 사용
optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(
    cost)  # tf.train.GradientDescentOptimizer(learning_rate) 말고 Adam이나 SGD도 사용가능
pred = tf.cast(A > 0.5, dtype=tf.float32)
accr = tf.reduce_mean(tf.cast(tf.equal(pred, Y), dtype=tf.float32))
training_epochs = 1000
batch_size = 100
display_step = 1

X_train = np.reshape(X_train, (-1, 3072))
X_test = np.reshape(X_test, (-1, 3072))
sess = tf.InteractiveSession()  # 그냥 session()하면 오류 나길래 저거 쓰면 오류 안난다고 해서...
sess.run(tf.global_variables_initializer())  # 얘도 그냥 쓰라고 해서...

num_batch = int(N / batch_size)
for epoch in range(training_epochs):
    avg_cost = 0
    for i in range(num_batch):
        randidx = np.random.randint(N, size=batch_size)

        batch_xs = X_train[randidx, :]
        batch_ys = Y_train[randidx, :]
        # 배치사이즈 = 100 설정 -> 미니배치 하겠다는 소리
        sess.run(optimizer, feed_dict={X: batch_xs, Y: batch_ys})
        avg_cost += sess.run(cost, feed_dict={X: batch_xs, Y: batch_ys}) / num_batch

    if epoch % display_step == 0:
        train_acc = accr.eval({X: batch_xs, Y: batch_ys})
        print("Epoch: %03d/%03d cost: %.9f train_acc: %.3f" % (epoch, training_epochs, avg_cost, train_acc))

h, c, a = sess.run([A, pred, accr], feed_dict={X: X_test, Y: Y_test})
print("\nHypothesis: ", h, "\nCorrect :", c, "\naccuracy : ", a)  # 내가 해봤을 땐 epoch 1000에서 정확도 97 이상
'''