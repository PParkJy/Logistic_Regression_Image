# Logistic Regression on Image data
### Introduction
The basic model of classification is logistic regression(LR).    
LR is the combination of linear regression and sigmoid activation function so, this model is mainly used in prediction or analysis.    
This project is focus on **classifying the multi class image**.    
The number of class is 5(bacteria, lateblight, targetspot, yellowleafcurl, healthy), I **used the one-vs-one, one-vs-rest method**.     
  - one-vs-one : 10 binary classifiers    
  - one-vs-rest : 5 binary classifiers    
  
I did performance evaluation with **accuracy, precision, recall, f1-score, AUC**. But, AUC is inaccurate.    
Also, the comparison in various situation(rotation, imbalance) is done.    
It showed performance of almost **80% or more in accuracy, f1-score, AUC**. 

<br/>

### Source code
Check the README.md of OVA, OVO directory.

<br/>

### Dataset
<a href="https://www.kaggle.com/saroz014/plant-diseases"> Original data </a>: Image data of plant diseases, 3GB.     
I preprocessed the **tomato data** of original as follows.
- size : 256x256 -> 32x32    
- filename : class_index_L<rotation angle>.jpg

<br/>

### Caution
The source codes of this project **aren't clean code**.    
So, you should use this project for **reference only**. 
