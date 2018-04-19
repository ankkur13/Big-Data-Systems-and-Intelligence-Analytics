import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import string
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
import re
from collections import Counter
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.corpus import stopwords
from wordcloud import WordCloud, STOPWORDS
from textblob import TextBlob
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import f1_score
from PIL import Image
import pickle
import boto
import boto.s3
import sys
import seaborn as sns
from boto.s3.key import Key
import matplotlib.pyplot as plt
#nltk.download('wordnet')


def main_function():
    data_ingestion()

#function to ingest the data into dataframe
def data_ingestion():
    df=pd.read_csv('Scrapped_content.csv',error_bad_lines=False)
    data_manipulation(df)

#function to manipulate the data into dataframe
def data_manipulation(df):
    columns=['TITLE','PUBLISHER','Content','Author','Category']
    df=df[columns]
    df=df.dropna(axis=0)
    df=df.reset_index()
    df=df.drop(['index'],axis=1)
    
    freq_words_title(df)
    freq_words_content(df)
    
    
#function to lemmatize the text    
def lemmatize_text(text):
    w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
    lemmatizer = nltk.stem.WordNetLemmatizer()
    return [lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)]

#function to stemmize the texts
def title_stemming(text):
    ps = PorterStemmer()
    l = []
    for i in text:
        for j in i:
            #converts list item to lower case
            p=j.lower()
            # removes punctuation,numbers and returns list of words
            q=re.sub('[^A-Za-z]+', ' ', p)
            l.append(ps.stem(q))
    return l
    
#function to find the most occuring words in the title column     
def freq_words_title(df):
    df['Title_lemmatized'] = df.TITLE.apply(lemmatize_text)
    lemmatization_title=df['Title_lemmatized']
    top_N = 100
    a=title_stemming(lemmatization_title)

    #remove all the stopwords from the text
    stop_words = list(get_stop_words('en'))         
    nltk_words = list(stopwords.words('english'))   
    stop_words.extend(nltk_words)

    words_to_remove=['thi','also','it ','us ','one',' ','. ','said','say','new','like','get','make','last','use','said ','new ','get ','say ','xbox ']
    filtered_sentence = []
    for w in a:
        if w not in (stop_words):
            if w not in (words_to_remove):
                filtered_sentence.append(w)

    # Remove characters which have length less than 2  
    without_single_chr = [word for word in filtered_sentence if len(word) > 2]

    # Remove numbers
    cleaned_data_title = [word for word in without_single_chr if not word.isnumeric()]        

    # Calculate frequency distribution
    word_dist = nltk.FreqDist(cleaned_data_title)
    rslt = pd.DataFrame(word_dist.most_common(top_N),columns=['Word', 'Frequency'])

    plt.figure(figsize=(15,10))
    sns.set_style("whitegrid")
    ax = sns.barplot(x="Word",y="Frequency", data=rslt.head(7))
    sentiments_title(df)
 
#function to calculate the sentiment on title column
def sentiments_title(df):
    bloblist_desc = list()

    df_review_str=df['TITLE'].astype(str)

    for row in df_review_str:
        blob = TextBlob(row)
        bloblist_desc.append((row,blob.sentiment.polarity, blob.sentiment.subjectivity))
        df_polarity_desc = pd.DataFrame(bloblist_desc, columns = ['TITLE','sentiment','polarity'])

    df_polarity_desc['Sentiment_Type'] = df_polarity_desc.apply(f_title, axis=1)
    
    plt.figure(figsize=(10,10))
    sns.set_style("whitegrid")
    ax = sns.countplot(x="Sentiment_Type", data=df_polarity_desc)
    machinelearning_title_column(df,df_polarity_desc)
    
#function to add the sentiment type in dataframe based on the condition
def f_title(df_polarity_desc):
    if df_polarity_desc['sentiment'] > 0:
        val = "Positive Title"
    elif df_polarity_desc['sentiment'] == 0:
        val = "Neutral Title"
    else:
        val = "Negative Title"
    return val

#function to calculate the most occuring words in the content column
def freq_words_content(df):
    top_N = 100
    df['Content_lemmatized'] = df.Content.apply(lemmatize_text)
    lemmatization_content=df['Content_lemmatized']
    a = title_stemming(lemmatization_content)
    
    #remove all the stopwords from the text
    stop_words = list(get_stop_words('en'))         
    nltk_words = list(stopwords.words('english'))   
    stop_words.extend(nltk_words)
    
    words_to_remove=['thi','also','it ','us ','one',' ','. ','said','say','new','like','get','make','last','use','said ']
    filtered_sentence = []
    for w in a:
        if w not in (stop_words):
            if w not in (words_to_remove):
                filtered_sentence.append(w)
    
    # Remove characters which have length less than 2  
    without_single_chr = [word for word in filtered_sentence if len(word) > 2]

    # Remove numbers
    cleaned_data_title = [word for word in without_single_chr if not word.isnumeric()]        

    # Calculate frequency distribution
    word_dist = nltk.FreqDist(cleaned_data_title)
    rslt = pd.DataFrame(word_dist.most_common(top_N), columns=['Word', 'Frequency'])

    plt.figure(figsize=(15,15))
    sns.set_style("whitegrid")
    ax = sns.barplot(x="Word",y="Frequency", data=rslt.head(7))
    sentiments_content(df)

#function to calculate the sentiment on the content column
def sentiments_content(df):
    bloblist_desc_content = list()

    df_review_str_content=df['Content'].astype(str)
    
    for row in df_review_str_content:
        blob_content = TextBlob(row)
        bloblist_desc_content.append((row,blob_content.sentiment.polarity, blob_content.sentiment.subjectivity))
        df_polarity_desc_content = pd.DataFrame(bloblist_desc_content, columns = ['Content','sentiment','polarity'])
    
    df_polarity_desc_content['Sentiment_Type'] = df_polarity_desc_content.apply(f_content, axis=1)
    
    plt.figure(figsize=(10,10))
    sns.set_style("whitegrid")
    ax = sns.countplot(x="Sentiment_Type", data=df_polarity_desc_content)
    
#function to add the sentiment type in dataframe based on the condition
def f_content(df_polarity_desc_content):
    if df_polarity_desc_content['sentiment'] > 0:
        val = "Positive Content"
    elif df_polarity_desc_content['sentiment'] == 0:
        val = "Neutral Content"
    else:
        val = "Negative Content"
    return val

#function to split and remove the stop words 
def text_process(title):
    nopunc=[word for word in title if word not in string.punctuation]
    nopunc=''.join(nopunc)
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

# Function to print and add metrics for News Sentiment
def print_metrics(df, model, recall_train, recall_test):
    df[model] = [float("{0:.5f}".format(recall_train)), float("{0:.5f}".format(recall_test))]
    return df

#function to print the metrics for News Category
def print_metrics_news_category(df, model, recall_train, recall_test):
    df[model] = [float("{0:.5f}".format(recall_train)), float("{0:.5f}".format(recall_test))]
    return df
    
#function to predict the sentiments of news article
def machinelearning_title_column(df,df_polarity_desc):
    # Category column could be transformed to numerical column
    category_dummies=pd.get_dummies(df['Category'])
    df_polarity_desc=pd.concat((df_polarity_desc,category_dummies),axis=1)

    title_class = df_polarity_desc[(df_polarity_desc['Sentiment_Type'] == 'Positive Title') | (df_polarity_desc['Sentiment_Type'] == 'Negative Title')]
    X_title=title_class['TITLE']
    title_class['Sentiment_Type'] = title_class.Sentiment_Type.map({'Positive Title':1, 'Negative Title':0})
    y=title_class['Sentiment_Type']
    
    bow_transformer=CountVectorizer(analyzer=text_process).fit(X_title)
    
    X_title = bow_transformer.transform(X_title)
    
    X_train, X_test, y_train, y_test = train_test_split(X_title, y, test_size=0.2, random_state=101)
    
    metrics_df = pd.DataFrame(index = ['Metrics_Train','Metrics_Test'])
    
    print('Naive Bayes Classifier Algorithm')
    nb = MultinomialNB()
    nb.fit(X_train, y_train)

    predict_train = nb.predict(X_train)
    predict_test = nb.predict(X_test)

    recall_train_nb=f1_score(y_train, predict_train)
    recall_test_nb=f1_score(y_test, predict_test)
    
    # Printing the training and testing metrices
    metrics_df = print_metrics(metrics_df, 'Naive Bayes_Model', recall_train_nb, recall_test_nb)
    print("The F-1 score metrics for Title column sentiment prediction is ", metrics_df)
    
    print('Multi Layer Perceptron Algorithm')
    # Import Multi-Layer Perceptron Classifier Model
    mlp = MLPClassifier(hidden_layer_sizes=(37,37,37))
    mlp.fit(X_train,y_train)

    predict_train = mlp.predict(X_train)
    predict_test_mlp = mlp.predict(X_test)

    recall_train_mlp=f1_score(y_train, predict_train)
    recall_test_mlp=f1_score(y_test, predict_test_mlp)
    
    # Printing the training and testing metrices
    metrics_df = print_metrics(metrics_df, 'MLP_Model', recall_train_mlp, recall_test_mlp)
    print("The F-1 score metrics for Title column sentiment prediction is ", metrics_df)
    
    print('XG Boost Classifier Algorithm')
    xgb=XGBClassifier()

    xgb_fit=xgb.fit(X_train, y_train)

    predict_train = xgb.predict(X_train)
    predict_test_xgb = xgb.predict(X_test)

    recall_train_xgb = f1_score(y_train, predict_train)
    recall_test_xgb = f1_score(y_test, predict_test_xgb)
    
    # Printing the training and testing metrices
    metrics_df = print_metrics(metrics_df, 'XGB_Model', recall_train_xgb, recall_test_xgb)
    print("The F-1 score metrics for Title column sentiment prediction is ", metrics_df)
    fun_pickle(nb,mlp,xgb)
    machinelearning_content_column(df_polarity_desc)
 

#function to predict the category of news article
def machinelearning_content_column(df_polarity_desc):
    
    df_news_business_entertainment=df_polarity_desc[(df_polarity_desc['Business']==1) | (df_polarity_desc['Entertainment']==1)]
    X_category=df_news_business_entertainment['TITLE']
    y=df_news_business_entertainment['Business']
    
    bow_transformer_category=CountVectorizer(analyzer=text_process).fit(X_category)
    
    X_category = bow_transformer_category.transform(X_category)
    X_train, X_test, y_train, y_test = train_test_split(X_category, y, test_size=0.1, random_state=101)
    
    print('Naive Bayes Classifier Algorithm')
    nb = MultinomialNB()
    nb.fit(X_train, y_train)

    predict_train = nb.predict(X_train)
    predict_test = nb.predict(X_test)

    recall_train_nb=f1_score(y_train, predict_train)
    recall_test_nb=f1_score(y_test, predict_test)
    
    metrics_df_category= pd.DataFrame(index = ['Metrics_Train','Metrics_Test'])
    
    metrics_df_category = print_metrics_news_category(metrics_df_category, 'Naive Bayes_Model', recall_train_nb, recall_test_nb)
    print('The F-1 score metrics for Content column sentiment prediction is ', metrics_df_category)
    
    print('XG Boost Classifier')
    xgb=XGBClassifier()
    xgb_fit=xgb.fit(X_train, y_train)

    predict_train = xgb.predict(X_train)
    predict_test = xgb.predict(X_test)
    
    recall_train_xgb = f1_score(y_train, predict_train)
    recall_test_xgb = f1_score(y_test, predict_test)
    
    # Printing the training and testing metrices

    metrics_df_category = print_metrics_news_category(metrics_df_category, 'XGB_Model', recall_train_xgb, recall_test_xgb)
    print('The F-1 score metrics for Content column sentiment prediction is ', metrics_df_category)
    
    print('Multi Layer Perceptron')
    mlp = MLPClassifier(hidden_layer_sizes=(37,37,37))
    mlp.fit(X_train,y_train)

    predict_train = mlp.predict(X_train)
    predict_test = mlp.predict(X_test)

    recall_train_mlp=f1_score(y_train, predict_train)
    recall_test_mlp=f1_score(y_test, predict_test)
    
    # Printing the training and testing metrices

    metrics_df_category = print_metrics_news_category(metrics_df_category, 'MLP_Model', recall_train_mlp, recall_test_mlp)
    print('The F-1 score metrics for Content column sentiment prediction is ', metrics_df_category)

#function to save the model to the disk
def fun_pickle(nb,mlp,xgb):
    # save the model to disk
    
    pickle_models=pd.DataFrame(columns=['Model Name','Model'])
    pickle_models=pickle_models.append({'Model Name':'Naive Bayes', 'Model': nb}, ignore_index=True)
    pickle_models=pickle_models.append({'Model Name':'Multi Layer Perceptron', 'Model': mlp}, ignore_index=True)
    pickle_models=pickle_models.append({'Model Name':'XG Boost', 'Model': xgb}, ignore_index=True)
    
    global filename
    filename = 'finalized_model_big_data.pkl'
    pickle.dump(pickle_models, open(filename, 'wb'))
 
#function to upload the saved file on the cloud
def uploadToS3(destinationPath, filePath, arg_AWSuser, arg_AWSpass):
    
    AWS_ACCESS_KEY_ID = arg_AWSuser
    AWS_SECRET_ACCESS_KEY = arg_AWSpass

    bucket_name = 'bigdatamodeldevelopmentdeployment'
    conn = boto.connect_s3(AWS_ACCESS_KEY_ID,AWS_SECRET_ACCESS_KEY)

    bucket = conn.create_bucket(bucket_name,location=boto.s3.connection.Location.DEFAULT)

    testfile = filePath
    print ('Uploading '+testfile+' to Amazon S3 bucket '+bucket_name)
    def percent_cb(complete, total):
        sys.stdout.write('.')
        sys.stdout.flush()

    k = Key(bucket)
    k.key = destinationPath+"/"+testfile
    k.set_contents_from_filename(testfile,cb=percent_cb, num_cb=10)

if __name__ == '__main__':
    main_function()
    
    arg_AWSuser = sys.argv[1]
    arg_AWSpass = sys.argv[2]
    
    uploadToS3("Models", filename, arg_AWSuser, arg_AWSpass)

