import pandas as pd
df=pd.read_csv('train.csv/fake_news_dataset.csv')
print(df.head())
print(df.info())
print(df.isnull().sum())
df=df.dropna()

from  sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

vectorizer= TfidfVectorizer()
x= vectorizer.fit_transform(df['text'])

y=df['label']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)
model = LogisticRegression()
model.fit(x_train, y_train)

accuracy = model.score(x_test,y_test)
print('Accuracy:',accuracy)
print('model Accuracy:',accuracy * 100 , '%')
sample = ['This news is fake and misleading']
sample_vector = vectorizer.transform(sample)

prediction = model.predict(sample_vector)
print('Prediction:',prediction)