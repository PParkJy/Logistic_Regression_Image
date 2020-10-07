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

json1 = load_json("./bacteria_lateblight/weight/50/bacteria_lateblight.json")
json2 = load_json("./bacteria_targetspot/weight/50/bacteria_targetspot.json")
json3 = load_json("./bacteria_yellowleafcurl/weight/50/bacteria_yellowleafcurl.json")
json4 = load_json("./healthy_bacteria/weight/50/healthy_bacteria.json")
json5 = load_json("./healthy_lateblight/weight/50/healthy_lateblight.json")
json6 = load_json("./healthy_targetspot/weight/50/healthy_targetspot.json")
json7 = load_json("./healthy_yellowleafcurl/weight/50/healthy_yellowleafcurl.json")
json8 = load_json("./lateblight_targetspot/weight/50/lateblight_targetspot.json")
json9 = load_json("./lateblight_yellowleafcurl/weight/50/lateblight_yellowleafcurl.json")
json10 = load_json("./targetspot_yellowleafcurl/weight/50/targetspot_yellowleafcurl.json")

model1 = load_weight(json1,"./bacteria_lateblight/weight/50/bacteria_lateblight.h5")
model2 = load_weight(json2,"./bacteria_targetspot/weight/50/bacteria_targetspot.h5")
model3 = load_weight(json3,"./bacteria_yellowleafcurl/weight/50/bacteria_yellowleafcurl.h5")
model4 = load_weight(json4,"./healthy_bacteria/weight/50/healthy_bacteria.h5")
model5 = load_weight(json5,"./healthy_lateblight/weight/50/healthy_lateblight.h5")
model6 = load_weight(json6,"./healthy_targetspot/weight/50/healthy_targetspot.h5")
model7 = load_weight(json7,"./healthy_yellowleafcurl/weight/50/healthy_yellowleafcurl.h5")
model8 = load_weight(json8,"./lateblight_targetspot/weight/50/lateblight_targetspot.h5")
model9 = load_weight(json9,"./lateblight_yellowleafcurl/weight/50/lateblight_yellowleafcurl.h5")
model10 = load_weight(json10,"./targetspot_yellowleafcurl/weight/50/targetspot_yellowleafcurl.h5")

print("load model from disk")

path1 = "./bacteria_lateblight/test/"
path2 = "./bacteria_targetspot/test/"
path3 = "./bacteria_yellowleafcurl/test/"
path4 = "./healthy_bacteria/test/"
path5 = "./healthy_lateblight/test/"
path6 = "./healthy_targetspot/test/"
path7 = "./healthy_yellowleafcurl/test/"
path8 = "./lateblight_targetspot/test/"
path9 = "./lateblight_yellowleafcurl/test/"
path10 = "./targetspot_yellowleafcurl/test/"
multi_path = "D:/jiyeon/AI/TomatoDiseases_imbalance/test/"

category1 = ["bacteria","lateblight"]
category2 = ["bacteria","targetspot"]
category3 = ["bacteria","yellowleafcurl"]
category4 = ["bacteria","healthy"]
category5 = ["lateblight","healthy"]
category6 = ["targetspot","healthy"]
category7 = ["yellowleafcurl","healthy"]
category8 = ["lateblight","targetspot"]
category9 = ["lateblight","yellowleafcurl"]
category10 = ["targetspot","yellowleafcurl"]
multi_category = ["bacteria", "healthy", "lateblight", "targetspot", "yellowleafcurl"]

X1_test = []; X2_test = []; X3_test = []; X4_test = []; X5_test = []
X6_test = []; X7_test = []; X8_test = []; X9_test = []; X10_test = []
Y1_test = []; Y2_test = []; Y3_test = []; Y4_test = []; Y5_test = []
Y6_test = []; Y7_test = []; Y8_test = []; Y9_test = []; Y10_test = []
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
set_label(X6_test,Y6_test,category6,path6)
set_label(X7_test,Y7_test,category7,path7)
set_label(X8_test,Y8_test,category8,path8)
set_label(X9_test,Y9_test,category9,path9)
set_label(X10_test,Y10_test,category10,path10)
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
X6_test = np.vstack(X6_test) / 256.0
Y6_test = np.array(Y6_test).astype('float32')
X7_test = np.vstack(X7_test) / 256.0
Y7_test = np.array(Y7_test).astype('float32')
X8_test = np.vstack(X8_test) / 256.0
Y8_test = np.array(Y8_test).astype('float32')
X9_test = np.vstack(X9_test) / 256.0
Y9_test = np.array(Y9_test).astype('float32')
X10_test = np.vstack(X10_test) / 256.0
Y10_test = np.array(Y10_test).astype('float32')
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
model6.compile(optimizer=optimizer, loss='binary_crossentropy',metrics=['accuracy',precision_threshold(0.5),recall_threshold(0.5)])
model7.compile(optimizer=optimizer, loss='binary_crossentropy',metrics=['accuracy',precision_threshold(0.5),recall_threshold(0.5)])
model8.compile(optimizer=optimizer, loss='binary_crossentropy',metrics=['accuracy',precision_threshold(0.5),recall_threshold(0.5)])
model9.compile(optimizer=optimizer, loss='binary_crossentropy',metrics=['accuracy',precision_threshold(0.5),recall_threshold(0.5)])
model10.compile(optimizer=optimizer, loss='binary_crossentropy',metrics=['accuracy',precision_threshold(0.5),recall_threshold(0.5)])

score1 = model1.evaluate(X1_test,Y1_test,verbose=0)
score2 = model2.evaluate(X2_test,Y2_test,verbose=0)
score3 = model3.evaluate(X3_test,Y3_test,verbose=0)
score4 = model4.evaluate(X4_test,Y4_test,verbose=0)
score5 = model5.evaluate(X5_test,Y5_test,verbose=0)
score6 = model6.evaluate(X6_test,Y6_test,verbose=0)
score7 = model7.evaluate(X7_test,Y7_test,verbose=0)
score8 = model8.evaluate(X8_test,Y8_test,verbose=0)
score9 = model9.evaluate(X9_test,Y9_test,verbose=0)
score10 = model10.evaluate(X10_test,Y10_test,verbose=0)

def result(classes,x_test,y_test,model,score,col):
    Y_test = y_test.argmax(axis=1)
    y_prob = model.predict_proba(x_test)
    y_prob = y_prob.argmax(axis=1)
    fpr, tpr, thresholds = roc_curve(Y_test,y_prob)

    precision1, recall1, _ = precision_recall_curve(Y_test, y_prob)
    plt.figure(0)
    plt.plot(recall1, precision1, color=col,lw=2,label=str(classes[0]) + '_' + str(
            classes[1]))
    plt.title('Recall-Precision Curve')
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.plot([0, 1], [0.5, 0.5], linestyle='--', color='gray')
    plt.legend(loc="lower left")
    #plt.show()
    plt.savefig('D:\\jiyeon\\AI\\image\\imbalance\\ovo\\50\\recall_precision\\' + str(classes[0])+'_'+str(classes[1])+ '.png')

    def plot_roc_curve(fpr,tpr):
        auc_score = auc(fpr, tpr)
        plt.figure(1)
        plt.plot(fpr,tpr, color=col,lw=2, label=str(classes[0]) + '_' + str(
            classes[1])+' (area = {0:0.2f}'.format(auc_score)+')')
        plt.plot([0, 1], [0, 1], linestyle='--',color='gray')
        plt.title('binary class ROC-curve')
        plt.xlabel('False positive Rate')
        plt.ylabel('Ture positive Rate')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.legend(loc="lower right")
        #plt.show()
        plt.savefig('D:\\jiyeon\\AI\\image\\imbalance\\ovo\\50\\roc_curve\\' + str(classes[0]) + '_' + str(
            classes[1]) + '.png')
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
result(category6,X6_test,Y6_test,model6,score6,'aqua')
result(category7,X7_test,Y7_test,model7,score7,'purple')
result(category8,X8_test,Y8_test,model8,score8,'black')
result(category9,X9_test,Y9_test,model9,score9,'darkorange')
result(category10,X10_test,Y10_test,model10,score10,'pink')

prob1 = model1.predict(X_multi,verbose=0)
prob2 = model2.predict(X_multi,verbose=0)
prob3 = model3.predict(X_multi,verbose=0)
prob4 = model4.predict(X_multi,verbose=0)
prob5 = model5.predict(X_multi,verbose=0)
prob6 = model6.predict(X_multi,verbose=0)
prob7 = model7.predict(X_multi,verbose=0)
prob8 = model8.predict(X_multi,verbose=0)
prob9 = model9.predict(X_multi,verbose=0)
prob10 = model10.predict(X_multi,verbose=0)

pred1 = model1.predict_classes(X_multi,verbose=0)
pred2 = model2.predict_classes(X_multi,verbose=0)
pred3 = model3.predict_classes(X_multi,verbose=0)
pred4 = model4.predict_classes(X_multi,verbose=0)
pred5 = model5.predict_classes(X_multi,verbose=0)
pred6 = model6.predict_classes(X_multi,verbose=0)
pred7 = model7.predict_classes(X_multi,verbose=0)
pred8 = model8.predict_classes(X_multi,verbose=0)
pred9 = model9.predict_classes(X_multi,verbose=0)
pred10 = model10.predict_classes(X_multi,verbose=0)

Y_temp = Y_multi
Y_multi = Y_multi.argmax(axis=1)
# 0 = bacteria
# 1 = healthy
# 2 = lateblight
# 3 = targetspot
# 4 = yellowleafcurl
multi_label = []

for i in range(len(pred1)):
    bacteria_cnt = 0
    healthy_cnt = 0
    lateblight_cnt = 0
    targetspot_cnt = 0
    yellowleafcurl_cnt = 0

    if(pred1[i] == 0): #bacteria
        pred1[i] = 0
        bacteria_cnt += max(prob1[i])
    else: #lateblight
        pred1[i] = 2
        lateblight_cnt += max(prob1[i])

    if(pred2[i] == 0): #bacteria
        pred2[i] = 0
        bacteria_cnt += max(prob2[i])
    else: #targetspot
        pred2[i] = 3
        targetspot_cnt += max(prob2[i])

    if(pred3[i] == 0): #bacteria
        pred3[i] = 0
        bacteria_cnt += max(prob3[i])
    else: #yellowleafcurl
        pred3[i] = 4
        yellowleafcurl_cnt += max(prob3[i])

    if(pred4[i] == 0): #bacteria
        pred4[i] = 0
        bacteria_cnt += max(prob4[i])
    else: #healthy
        pred4[i] = 1
        healthy_cnt += max(prob4[i])

    if(pred5[i] == 0): #lateblight
        pred5[i] = 2
        lateblight_cnt += max(prob5[i])
    else: #healthy
        pred5[i] = 1
        healthy_cnt += max(prob5[i])

    if(pred6[i] == 0): #targetspot
        pred6[i] = 3
        targetspot_cnt += max(prob6[i])
    else: #healthy
        pred6[i] = 1
        healthy_cnt += max(prob6[i])

    if(pred7[i] == 0): #yellowleafcurl
        pred7[i] = 4
        yellowleafcurl_cnt += max(prob7[i])
    else: #healthy
        pred7[i] = 1
        healthy_cnt += max(prob7[i])

    if(pred8[i] == 0): #lateblight
        pred8[i] = 2
        lateblight_cnt += max(prob8[i])
    else: #targetspot
        pred8[i] = 3
        targetspot_cnt += max(prob8[i])

    if(pred9[i] == 0): #lateblight
        pred9[i] = 2
        lateblight_cnt += max(prob9[i])
    else: #yellowleafcurl
        pred9[i] = 4
        yellowleafcurl_cnt += max(prob9[i])

    if(pred10[i] == 0): #targetspot
        pred10[i] = 3
        targetspot_cnt += max(prob10[i])
    else: #yellowleafcurl
        pred10[i] = 4
        yellowleafcurl_cnt += max(prob10[i])

    temp_list = [bacteria_cnt,healthy_cnt,lateblight_cnt,targetspot_cnt,yellowleafcurl_cnt]
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
    print(multi_category[i])
    plt.plot(fpr[i], tpr[i], color=color, lw=lw, label='ROC curve of '+str(multi_category[i])+' (area = {0:0.2f}'.format(roc_auc[i])+') ')

plt.plot([0, 1], [0, 1], 'k--', lw=lw,color='gray')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Tomato disease multi class ROC-curve')
plt.legend(loc="lower right")
#plt.show()
plt.savefig(
    'D:\\jiyeon\\AI\\image\\imbalance\\ovo\\50\\multi_roc\\multi_roc_curve.png')
print('========================================================================================')
print(confusion_matrix(Y_multi, multi_label))
print(classification_report(Y_multi, multi_label, target_names=multi_category))
print('=========================================================================================')