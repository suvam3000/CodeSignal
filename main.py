from io import StringIO
import pandas as pd
import requests
import numpy as np
import re
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


data_url = "https://abnormalsecurity-public.s3.us-east-1.amazonaws.com/url_interview_cleaned.csv"
content = requests.get(data_url).content.decode('utf-8')
container = StringIO(content)
df = pd.read_csv(container, sep='\t', names=['label', 'url'])
good_urls = list(df[df['label'] == "0"]['url'])
bad_urls = list(df[df['label'] == "1"]['url'])


my_dict = {}

for i in good_urls:
    my_dict[i] = 0
    
for i in bad_urls:
    my_dict[i] = 1

# Take a look at some of the data.
# print('BAD:')
# print(bad_urls[1:10])
# print('GOOD:')
# print(good_urls[1:10])



def getTokens(input):
    tokensBySlash = str(input.encode('utf-8')).split('/')
    allTokens=[]
    for i in tokensBySlash:
        tokens = str(i).split('-')
        tokensByDot = []
        for j in range(0,len(tokens)):
            tempTokens = str(tokens[j]).split('.')
            tokentsByDot = tokensByDot + tempTokens
        allTokens = allTokens + tokens + tokensByDot
    allTokens = list(set(allTokens))
    if 'com' in allTokens:
        allTokens.remove('com')
    return allTokens


y_good = [0]*len(good_urls)
y_bad = [1]*len(bad_urls)

y = y_good + y_bad
myUrls = good_urls + bad_urls

indexing = [ i for i in range(len(y))]





vectorizer = TfidfVectorizer( tokenizer=getTokens ,use_idf=True, smooth_idf=True, sublinear_tf=False)
features = vectorizer.fit_transform(myUrls).toarray()
labels = y
features.shape

model = LogisticRegression(max_iter=1000,random_state=0)
X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels, indexing, test_size=0.20, random_state=0)
model.fit(X_train, y_train)
y_pred_proba = model.predict_proba(X_test)
y_pred = model.predict(X_test)
clf = LogisticRegression(random_state=0) 
clf.fit(X_train,y_train)
train_score = clf.score(X_train, y_train)
test_score = clf.score(X_test, y_test)
# print (f'train_score : {train_score}')
# print (f'test_score : {test_score}')

def inner_predict(input_url):
    X_predict = [input_url]
    X_predict = vectorizer.transform(X_predict)
    y_Predict = clf.predict(X_predict)
    
    is_bad =  re.search('1$', str(y_Predict)[-2:-1])
    
    if is_bad is not None:  # True if the URL is malicious (bad urls lable 1)
        return True
    else:
        return False
        

def classify_url(url):
    
    return inner_predict(url)
