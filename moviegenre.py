import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
# preprocessing of dataset from text to csv
with open('train_data.txt', 'r', encoding='utf-8') as file:
    raw_data = file.readlines()
split_data = [entry.strip().split(' ::: ') for entry in raw_data]
df = pd.DataFrame(split_data, columns=['ID', 'Title','Genre','Plot_Summary'])
df.to_csv('train_data.csv', index=False)
with open('test_data.txt', 'r', encoding='utf-8') as file:
    raw_data = file.readlines()
split_data = [entry.strip().split(' ::: ') for entry in raw_data]
df = pd.DataFrame(split_data, columns=['ID', 'Title','Plot_Summary'])
df.to_csv('test_data.csv', index=False)

# load datasets
print("MOVIE GENRE CLASSIFICATION ")
train_data = pd.read_csv('train_data.csv')
test_data = pd.read_csv('test_data.csv')
print(train_data.columns)

X_train = train_data.iloc[:, 3]  
y_train = train_data['Genre']
X_test = test_data.iloc[:, 2]  

tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)
nb_classifier = MultinomialNB()

# Train the classifier
nb_classifier.fit(X_train_tfidf, y_train)

# Predict the test data
y_pred = nb_classifier.predict(X_test_tfidf)
test_data['predicted_genre'] = y_pred
accuracy = accuracy_score(test_data['predicted_genre'], y_pred)
print("Accuracy:", accuracy)

# Display classification report
print("Classification Report:")
print(classification_report(test_data['predicted_genre'], y_pred))

# Display confusion matrix
conf_matrix = confusion_matrix(test_data['predicted_genre'], y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="Blues", xticklabels=nb_classifier.classes_, yticklabels=nb_classifier.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
# Bar plot for distribution of predicted genres
plt.figure(figsize=(10, 6))
sns.countplot(x='predicted_genre', data=test_data, order=test_data['predicted_genre'].value_counts().index, palette='viridis')
plt.title('Distribution of Predicted Genres')
plt.xlabel('Genre')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# Print the test data with predicted genres
print("THE PREDICTED RESULTS:")
print(test_data[['ID', 'Title', 'Plot_Summary', 'predicted_genre']])
