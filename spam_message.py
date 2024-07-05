#Import libraries
import pandas as pd# data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np# linear algebra
import matplotlib.pyplot as plt# for data visualization purposes
from sklearn.feature_extraction.text import CountVectorizer
import seaborn as sns # for statistical data visualization

#Import dataset
data=pd.read_csv('spam_or_not_spam.csv')
data.head(10)# preview the dataset

# drop Nan value from dataset
data.dropna(inplace=True)
data.head(10)

# encode remaining variables with TF-IDF encoding
# convert text variable to numeric
from sklearn.feature_extraction.text import TfidfVectorizer
tf_idf= TfidfVectorizer()
train_data = tf_idf.fit_transform(data["email"]).toarray()
label_data = data["label"]

print(train_data)

# split X and y into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train_data, label_data, test_size = 0.20, random_state = 10)

# Dimensional All Data
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# percentage for spam and ham
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes=True)
plt.figure(figsize = (6,5))
ax = sns.countplot(x='label',data=data)

# train a Gaussian Naive Bayes classifier on the training set
from sklearn.naive_bayes import GaussianNB
# instantiate the model
model = GaussianNB()
# fit the model
model.fit(X_train,y_train)

# predict new data
y_pred = model.predict(X_test)
y_pred[:100]

# Evaluate the model
from sklearn.metrics import accuracy_score , precision_score , recall_score
#  classification accuracy
acc = accuracy_score(y_pred , y_test)
print("Accuracy  = ",acc) # tp+fn / (tp+tn+fp+fn)

# classification precision
precision = precision_score(y_pred , y_test)
print("Precision = ",precision) # tp / (tp+fp)

# classification recall
recall = recall_score(y_pred , y_test)
print("Recall = ",recall) # tp / (tp+fn)

# Print the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# visualize confusion matrix
import matplotlib.pyplot as plt
import numpy
from sklearn import metrics
confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [0, 1])
cm_display.plot()
plt.show()

# Classification metrices
from sklearn.metrics import classification_report
target_names = ['class 0', 'class 1']
print(classification_report(y_test, y_pred, target_names=target_names))