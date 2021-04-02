import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


df = pd.read_csv('./SMSSpamCollection', sep='\t')
df.columns = ['Label', 'Message']

# make strings *nicer*
df['Message'] = df['Message'].str.lower()
df['Message'] = df['Message'].str.replace('[^\w\s]', '')

test_data = df[::10]
train_data = df[df.index % 10 != 0]

train_labels = train_data['Label']
train_features = train_data['Message']

# assign occurences of words to an index
vectorizer = CountVectorizer(analyzer='word')
vectorized_train_features = vectorizer.fit_transform(train_features)

# change to frequencies to avoid overvaluing common words
tf_transformer = TfidfTransformer(use_idf=False)
transformed_train_features = tf_transformer.fit_transform(vectorized_train_features)

classifier = MultinomialNB(alpha = 0.2).fit(transformed_train_features, train_labels)


test_labels = train_data['Label']
test_features = train_data['Message']

vectorized_test_features = vectorizer.transform(test_features)
transformed_test_features = tf_transformer.transform(vectorized_test_features)

prediction = classifier.predict(transformed_test_features)

#includes accuracy, precision, recall, F1
class_report = classification_report(test_labels, prediction, target_names=['ham', 'spam'])
print("\nReport: \n", class_report)

#includes confusion matrix
confusion = confusion_matrix(test_labels, prediction)
print("Confusion Matrix: \n", confusion)
