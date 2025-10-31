import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

data_dict = pickle.load(open('dataset.pickle', 'rb'))

#Reloading the preprocessed dataset stored locally in pickle file format
data = np.asarray(data_dict['dataset'])
labels = np.asarray(data_dict['labels'])

#Splitting the dataset into training and testing set using train_test_split from sklearn library
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

model = RandomForestClassifier(n_estimators=100)

#Training the model using RandomForestClassifier
model.fit(X_train, y_train)

#Predicting labels for the test dataset using the pretrained model.
y_pred = model.predict(X_test)

score = accuracy_score(y_pred, y_test)

print('{}% of samples were classified correctly !'.format(score * 100))
print("Classification Report : \n", classification_report(y_test, y_pred))

# Saving the model for using in real_time testing.
f = open('model.p', 'wb')
pickle.dump({'model' : model}, f)
f.close()