from flask import Flask, render_template, url_for, request
import pandas as pd
import numpy as np
from nltk.stem.porter import PorterStemmer
import re
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


## funtions for cleaning

def remove_pattern(input_txt,pattern):
    r = re.findall(pattern,input_txt)
    for i in r:
        input_txt = re.sub(i,'',input_txt)
    return input_txt
def count_punct(text):
    count = sum([1 for char in text if char in string.punctuation])
    return round(count/(len(text) - text.count(" ")),3)*100


app = Flask(__name__)


data = pd.read_csv("sentiment.tsv",sep = '\t')
data.columns = ["label","body_text"]
# Features and Labels
data['label'] = data['label'].map({'pos': 1, 'neg': 0})
data['tidy_tweet'] = np.vectorize(remove_pattern)(data['body_text'],"@[\w]*")
tokenized_tweet = data['tidy_tweet'].apply(lambda x: x.split())
stemmer = PorterStemmer()
tokenized_tweet = tokenized_tweet.apply(lambda x: [stemmer.stem(i) for i in x])
for i in range(len(tokenized_tweet)):
    tokenized_tweet[i] = ' '.join(tokenized_tweet[i])
data['tidy_tweet'] = tokenized_tweet
data['body_len'] = data['body_text'].apply(lambda x:len(x) - x.count(" "))
data['punct%'] = data['body_text'].apply(lambda x:count_punct(x))
X = data['tidy_tweet']
y = data['label']
print(type(X))
# Extract Feature With CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(X) # Fit the Data
X = pd.concat([data['body_len'],data['punct%'],pd.DataFrame(X.toarray())],axis = 1)

## Using Classifier
clf = MultinomialNB()
clf.fit(X,y)


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict/',methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = pd.DataFrame(cv.transform(data).toarray())
        body_len = pd.DataFrame([len(data) - data.count(" ")])
        punct = pd.DataFrame([count_punct(data)])
        total_data = pd.concat([body_len,punct,vect],axis = 1)
        my_prediction = clf.predict(total_data)
        score=clf.predict_proba(total_data)[:, 1]

        # using dataframe
        result = data
        result = pd.DataFrame(result, columns=["text"])
        result["sentiment-type"] = my_prediction
        result["sentiment-type"] = result['sentiment-type'].map({1: "positive", 0: "negative"})
        result["probability"] = score
        json_table = result.to_json(orient='records')
    return app.response_class(
        response=json_table,
        status=200,
        mimetype='application/json'
    )


if __name__ == '__main__':
    app.run()
