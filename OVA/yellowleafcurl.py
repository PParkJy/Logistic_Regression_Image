import os, cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras import backend as K
import matplotlib.pyplot as plt
from keras.optimizers import Adam
train_path = './yellowleafcurl/train/'
test_path = './yellowleafcurl/test/'
classes = ["yellowleafcurl", "non_yellowleafcurl"]

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
            #print(paths + filename)
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

print(X_train.shape)  # (11896,3072) -> 11896 = trainset의 개수 (bacteria : 6808, healthy: 5088), 3072 = 특징의 개수 = 32(=이미지 너비) * 32(=이미지 높이) * 3(=RGB값)
print(Y_train.shape)  # (11896,2) -> 2 = label의 개수(bacteria, healthy)
print(X_test.shape)  # (2972.3072) ->2972 = trainset의 개수
print(Y_test.shape)  # (2972,2)

def logistic_model(input_dim, output_dim):
    model = Sequential()
    model.add(Dense(output_dim, input_dim=input_dim,activation='sigmoid'))
    return model

batch_size = 100
epoch = 50
num_label =2
input_dim = 32*32*3

model = logistic_model(input_dim,num_label)
model.summary()

def precision_threshold(threshold=0.5):
    def precision(y_true, y_pred):
        """Precision metric.
        Computes the precision over the whole batch using threshold_value.
        """
        threshold_value = threshold
        # Adaptation of the "round()" used before to get the predictions. Clipping to make sure that the predicted raw values are between 0 and 1.
        y_pred = K.cast(K.greater(K.clip(y_pred, 0, 1), threshold_value), K.floatx())
        # Compute the number of true positives. Rounding in prevention to make sure we have an integer.
        true_positives = K.round(K.sum(K.clip(y_true * y_pred, 0, 1)))
        # count the predicted positives
        predicted_positives = K.sum(y_pred)
        # Get the precision ratio
        precision_ratio = true_positives / (predicted_positives + K.epsilon())
        return precision_ratio
    return precision

def recall_threshold(threshold = 0.5):
    def recall(y_true, y_pred):
        """Recall metric.
        Computes the recall over the whole batch using threshold_value.
        """
        threshold_value = threshold
        # Adaptation of the "round()" used before to get the predictions. Clipping to make sure that the predicted raw values are between 0 and 1.
        y_pred = K.cast(K.greater(K.clip(y_pred, 0, 1), threshold_value), K.floatx())
        # Compute the number of true positives. Rounding in prevention to make sure we have an integer.
        true_positives = K.round(K.sum(K.clip(y_true * y_pred, 0, 1)))
        # Compute the number of positive targets.
        possible_positives = K.sum(K.clip(y_true, 0, 1))
        recall_ratio = true_positives / (possible_positives + K.epsilon())
        return recall_ratio
    return recall

optimizer = Adam(lr=0.0001)
model.compile(optimizer=optimizer, loss='binary_crossentropy',metrics=['accuracy',precision_threshold(0.5),recall_threshold(0.5)]) #optimizer를 adam으로도 바꿔보자
history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=epoch, verbose=1)

plt.plot(history.history['loss'],label='loss')
plt.xlabel('epoch')
plt.title('yellowleafcurl')
plt.legend(loc='lower right')
plt.show()
plt.plot(history.history['acc'],label='accuracy')
plt.xlabel('epoch')
plt.legend(loc='lower right')
plt.show()
plt.plot(history.history['precision'],label='precision')
plt.xlabel('epoch')
plt.legend(loc='lower right')
plt.show()
plt.plot(history.history['recall'],label='recall')
plt.xlabel('epoch')
plt.legend(loc='lower right')
plt.show()

json_string = model.to_json()
open('./yellowleafcurl/weight/'+str(epoch)+'/yellowleafcurl.json','w').write(json_string)
model.save_weights('./yellowleafcurl/weight/'+str(epoch)+'/yellowleafcurl.h5')
print('model save')