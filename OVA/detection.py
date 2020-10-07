import os
import cv2
import numpy as np
from keras.models import model_from_json
from keras import backend as K
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,roc_auc_score,auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report, confusion_matrix
from keras.optimizers import Adam
from scipy import interp
from itertools import cycle
from keras.utils import to_categorical

def load_json(path):
    json_file = open(path,"r")
    loaded_model_json = json_file.read()
    json_file.close()
    return loaded_model_json

def load_weight(js,path):
    loaded_model = model_from_json(js)
    loaded_model.load_weights(path)
    return loaded_model

json1 = load_json("./bacteria/weight/50/bacteria.json")
json2 = load_json("./healthy/weight/50/healthy.json")
json3 = load_json("./lateblight/weight/50/lateblight.json")
json4 = load_json("./targetspot/weight/50/targetspot.json")
json5 = load_json("./yellowleafcurl/weight/50/yellowleafcurl.json")

model1 = load_weight(json1,"./bacteria/weight/50/bacteria.h5")
model2 = load_weight(json2,"./healthy/weight/50/healthy.h5")
model3 = load_weight(json3,"./lateblight/weight/50/lateblight.h5")
model4 = load_weight(json4,"./targetspot/weight/50/targetspot.h5")
model5 = load_weight(json5,"./yellowleafcurl/weight/50/yellowleafcurl.h5")

print("load model from disk")

path1 = "./bacteria/test/"
path2 = "./healthy/test/"
path3 = "./lateblight/test/"
path4 = "./targetspot/test/"
path5 = "./yellowleafcurl/test/"

multi_path = "D:/jiyeon/AI/TomatoDiseases_imbalance/test/"

category1 = ["bacteria","non_bacteria"]
category2 = ["healthy","non_healthy"]
category3 = ["lateblight","non_lateblight"]
category4 = ["targetspot","non_targetspot"]
category5 = ["yellowleafcurl","non_yellowleafcurl"]

multi_category = ["bacteria", "healthy", "lateblight", "targetspot", "yellowleafcurl"]

X1_test = []; X2_test = []; X3_test = []; X4_test = []; X5_test = []
Y1_test = []; Y2_test = []; Y3_test = []; Y4_test = []; Y5_test = []
X_multi = []; Y_multi = []

width = 32
height = 32

def load_image(paths, x_list, y_list, l):
    for top, dir, f in os.walk(paths):
        for filename in f:
            #print(paths + filename)
            img = cv2.imread(paths + filename)
            img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA).flatten()
            x_list.append([np.array(img).astype('float32')])
            y_list.append(l)

def set_label(x,y,classes,path):
    for idx, label in enumerate(classes):
        labeling = [0 for i in range(len(classes))]
        #print(idx) -> 0 1 원핫인코딩
        labeling[idx] = 1
        #print(label) -> bacteria [1, 0] healthy [0,1]
        #print(labeling)
        test_image = path + label + "/"
        load_image(test_image, x, y, labeling)  # test 이미지 로드

set_label(X1_test,Y1_test,category1,path1)
set_label(X2_test,Y2_test,category2,path2)
set_label(X3_test,Y3_test,category3,path3)
set_label(X4_test,Y4_test,category4,path4)
set_label(X5_test,Y5_test,category5,path5)
set_label(X_multi,Y_multi,multi_category,multi_path)

print('load image')

X1_test = np.vstack(X1_test) / 256.0
Y1_test = np.array(Y1_test).astype('float32')
X2_test = np.vstack(X2_test) / 256.0
Y2_test = np.array(Y2_test).astype('float32')
X3_test = np.vstack(X3_test) / 256.0
Y3_test = np.array(Y3_test).astype('float32')
X4_test = np.vstack(X4_test) / 256.0
Y4_test = np.array(Y4_test).astype('float32')
X5_test = np.vstack(X5_test) / 256.0
Y5_test = np.array(Y5_test).astype('float32')
X_multi = np.vstack(X_multi) / 256.0
Y_multi = np.array(Y_multi).astype('float32')

def precision_threshold(threshold=0.5):
    def precision(y_true, y_pred):
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

optimizer = Adam(lr=0.001)
model1.compile(optimizer=optimizer, loss='binary_crossentropy',metrics=['accuracy',precision_threshold(0.5),recall_threshold(0.5)])
model2.compile(optimizer=optimizer, loss='binary_crossentropy',metrics=['accuracy',precision_threshold(0.5),recall_threshold(0.5)])
model3.compile(optimizer=optimizer, loss='binary_crossentropy',metrics=['accuracy',precision_threshold(0.5),recall_threshold(0.5)])
model4.compile(optimizer=optimizer, loss='binary_crossentropy',metrics=['accuracy',precision_threshold(0.5),recall_threshold(0.5)])
model5.compile(optimizer=optimizer, loss='binary_crossentropy',metrics=['accuracy',precision_threshold(0.5),recall_threshold(0.5)])

score1 = model1.evaluate(X1_test,Y1_test,verbose=0)
score2 = model2.evaluate(X2_test,Y2_test,verbose=0)
score3 = model3.evaluate(X3_test,Y3_test,verbose=0)
score4 = model4.evaluate(X4_test,Y4_test,verbose=0)
score5 = model5.evaluate(X5_test,Y5_test,verbose=0)

def result(classes,x_test,y_test,model,score,col):
    Y_test = y_test.argmax(axis=1)
    y_prob = model.predict_proba(x_test)
    y_prob = y_prob.argmax(axis=1)
    fpr, tpr, thresholds = roc_curve(Y_test,y_prob)

    precision1, recall1, _ = precision_recall_curve(Y_test, y_prob)
    plt.figure(0)
    plt.plot(recall1, precision1,lw=2,label=str(classes[0]), color=col)
    plt.title('Recall-Precision Curve')
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.plot([0, 1], [0.5, 0.5], linestyle='--',color='gray')
    plt.legend(loc="lower left")
    #plt.show()
    plt.savefig('D:\\jiyeon\\AI\\image\\imbalance\\ova\\50\\recall_precision\\' + str(classes[0]) + '.png')
    def plot_roc_curve(fpr,tpr):
        auc_score = auc(fpr, tpr)
        plt.figure(1)
        plt.plot(fpr,tpr, color=col,lw=2, label=str(classes[0]) + '_' + 'area = {0:0.2f}'.format(auc_score))
        plt.plot([0, 1], [0, 1], linestyle='--',color='gray')
        plt.title('binary class ROC-curve')
        plt.xlabel('False positive Rate')
        plt.ylabel('Ture positive Rate')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.legend(loc="lower right")
        #plt.show()
        plt.savefig('D:\\jiyeon\\AI\\image\\imbalance\\ova\\50\\roc_curve\\' + str(classes[0])+ '.png')
    plot_roc_curve(fpr,tpr)
    print('========================================================================================')
    print('Test score: ', score[0], 'accuracy: ',score[1], 'precision: ', score[2], 'recall: ',score[3], 'f1-score: ', 2*((score[2]*score[3])/(score[2]+score[3]+K.epsilon())))
    print(confusion_matrix(Y_test,y_prob))
    print(classification_report(Y_test, y_prob, target_names=classes))
    print('=========================================================================================')

result(category1,X1_test,Y1_test,model1,score1,'red')
result(category2,X2_test,Y2_test,model2,score2,'orange')
result(category3,X3_test,Y3_test,model3,score3,'yellow')
result(category4,X4_test,Y4_test,model4,score4,'green')
result(category5,X5_test,Y5_test,model5,score5,'blue')

prob1 = model1.predict(X_multi,verbose=0)
prob2 = model2.predict(X_multi,verbose=0)
prob3 = model3.predict(X_multi,verbose=0)
prob4 = model4.predict(X_multi,verbose=0)
prob5 = model5.predict(X_multi,verbose=0)

Y_temp = Y_multi
Y_multi = Y_multi.argmax(axis=1)
# 0 = bacteria
# 1 = healthy
# 2 = lateblight
# 3 = targetspot
# 4 = yellowleafcurl
multi_label = []

for i in range(len(Y_multi)):
    temp_list = [prob1[i][0],prob2[i][0],prob3[i][0],prob4[i][0],prob5[i][0]]
    print(temp_list)
    temp = max(temp_list)
    idx = temp_list.index(temp)

    if(idx == 0): #bacteria
        multi_label.append(0)
    elif(idx == 1): #healthy
        multi_label.append(1)
    elif(idx == 2): #lateblight
        multi_label.append(2)
    elif(idx == 3): #targerspot
        multi_label.append(3)
    else: #yellowleafcurl
        multi_label.append(4)
print(multi_label)

print('========================================================================================')
print(confusion_matrix(Y_multi, multi_label))
print(classification_report(Y_multi, multi_label, target_names=multi_category))
print('=========================================================================================')

fpr = dict()
tpr = dict()
roc_auc = dict()
multi_temp = to_categorical(multi_label)
for i in range(len(multi_category)):
    fpr[i], tpr[i], _ = roc_curve(Y_temp[:,i], multi_temp[:,i])
    roc_auc[i] = auc(fpr[i], tpr[i])
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(len(multi_category))]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(len(multi_category)):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= len(multi_category)

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
plt.figure(2)
lw = 2

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue','red','green'])
for i, color in zip(range(len(multi_category)), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,label='ROC curve of '+str(multi_category[i])+' (area = {0:0.2f}'.format(roc_auc[i])+') ')

plt.plot([0, 1], [0, 1], 'k--', lw=lw,color='gray')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Tomato disease multi class ROC-curve')
plt.legend(loc="lower right")
#plt.show()
plt.savefig(
    'D:\\jiyeon\\AI\\image\\imbalance\\ova\\50\\multi_roc\\multi_roc_curve.png')
print('========================================================================================')
print(confusion_matrix(Y_multi, multi_label))
print(classification_report(Y_multi, multi_label, target_names=multi_category))
print('=========================================================================================')