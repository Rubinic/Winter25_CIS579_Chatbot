from textblob.classifiers import NaiveBayesClassifier
from textblob.classifiers import DecisionTreeClassifier
from textblob import TextBlob
import joblib
from sklearn.metrics import classification_report, confusion_matrix


import pandas as pd


## Load first dataset
sentiment_df = pd.read_csv("Datasets/MentalHealthDataset.csv")

## Cleaning first dataset
posneg = {'negative': 'neg', 'positive': 'pos', 'very negative': 'neg'}
sentiment_df['posneg'] = sentiment_df['predicted'].map(posneg)
sentiment_df = sentiment_df.drop(['predicted', 'intensity'], axis=1)
sentiment_df = sentiment_df.dropna(subset=['posneg'])


# NaiveBayesClassifier
## creat testing and training datasets
training_data = sentiment_df.sample(frac = 0.8, random_state=150)
testing_data = sentiment_df.drop(training_data.index)

# NaiveBayesClassifier
## save out training and testing datafiles
training_data.to_csv('TestTrainData/training_data.csv', index=False, header=False)
testing_data.to_csv('TestTrainData/testing_data.csv', index=False, header = False)

## creating model based on training data
with open('TestTrainData/training_data.csv', 'r') as trainingdata:
    nb_cl = NaiveBayesClassifier(trainingdata, format="csv")


## Accuracy score of current model
with open('TestTrainData/testing_data.csv', 'r') as testingdata:
    nb_score = nb_cl.accuracy(testingdata, format='csv')
    print("NAIVE BAYES CLASSIFIER PERFORMANCE")
    print("Accuracy Score: ", nb_score)


## Confusion Matrix and Classification Report
testing_df = pd.read_csv('TestTrainData/testing_data.csv', names = ['Post', 'Class'])
testing_df['Pred_Class'] = testing_df['Post'].apply(lambda x: nb_cl.classify(x))

print("Classification Report:")
print(classification_report(testing_df['Class'], testing_df['Pred_Class']))

print("Confusion Matrix:")
print(confusion_matrix(testing_df['Class'], testing_df['Pred_Class']))



# DecisionTreeClassifier
## tokenizing data before passing through decisiontree
training_data['token_sentence'] = training_data['posts'].str.split()
testing_data['token_sentence'] = testing_data['posts'].str.split()

## save out training and testing datafiles
training_data.drop(['posts'], axis=1).reindex(columns= ['token_sentence', 'posneg']).to_csv('TestTrainData/training_data_dt.csv', index=False, header=False)
testing_data.drop(['posts'], axis=1).reindex(columns= ['token_sentence', 'posneg']).to_csv('TestTrainData/testing_data_dt.csv', index=False, header = False)


## creating model based on training data
with open('TestTrainData/training_data_dt.csv', 'r') as trainingdata:
    dt_cl = DecisionTreeClassifier(trainingdata, format="csv")


## Accuracy score of current model
with open('TestTrainData/testing_data_dt.csv', 'r') as testingdata:
    dt_score = dt_cl.accuracy(testingdata, format='csv')
    print("DECISION TREE CLASSIFIER")
    print("Accuracy Score: ", dt_score)

## Confusion Matrix and Classification Report
testing_df = pd.read_csv('TestTrainData/testing_data_dt.csv', names = ['Post', 'Class'])
testing_df['Pred_Class'] = testing_df['Post'].apply(lambda x: dt_cl.classify(x))

print("Classification Report:")
print(classification_report(testing_df['Class'], testing_df['Pred_Class']))

print("Confusion Matrix:")
print(confusion_matrix(testing_df['Class'], testing_df['Pred_Class']))



# Incorporating in supplemental dataset
## Reading in csv
second_df = pd.read_csv("Datasets/socialmediasentimentdataset.csv")

## Cleaning second dataset
posneg = {'Negative': 'neg', 'Positive': 'pos'}
second_df['posneg'] = second_df['Sentiment'].map(posneg)
second_df = second_df.drop(['Hashtags', 'Sentiment'], axis=1)
second_df = second_df.dropna(subset=['posneg'])


### UPDATING DATA TO MODEL AND SAVING OUT MODEL WITH BEST PERFORMANCE FOR CHATBOT  ###
if (nb_score > dt_score):
    ## adding in supplemental data for model
    sup_data = list(zip(second_df['Text'], second_df['posneg']))
    nb_cl_2 = nb_cl
    nb_cl_2.update(sup_data)

    ## Getting new accuracy score with new data included.
    with open('TestTrainData/testing_data.csv', 'r') as testingdata:
        nb_score_2 = nb_cl_2.accuracy(testingdata, format='csv')
        print("NAIVE BAYES CLASSIFIER PERFORMANCE")
        print("Accuracy Score: ", nb_score_2)

    if (nb_score_2 > nb_score):
        joblib.dump(nb_cl_2, 'BestClassifierModel.pkl')
    elif (nb_score_2 < nb_score):
        joblib.dump(nb_cl, 'BestClassifierModel.pkl')

    print("Script execution completed successfully.")
    quit()

elif (nb_score < dt_score):
    ## adding in supplemental data for model
    sup_data = list(zip(second_df['Text'], second_df['posneg']))
    dt_cl_2 = nb_cl
    dt_cl_2.update(sup_data)

    ## Getting new accuracy score with new data included.
    with open('TestTrainData/testing_data.csv', 'r') as testingdata:
        dt_score_2 = dt_cl_2.accuracy(testingdata, format='csv')
        print("DECISION TREE CLASSIFIER PERFORMANCE")
        print("Accuracy Score: ", dt_score_2)

    if (dt_score_2 > dt_score):
        joblib.dump(dt_cl_2, 'BestClassifierModel.pkl')
    elif (dt_score_2 < dt_score):
        joblib.dump(dt_cl, 'BestClassifierModel.pkl')

    print("Script execution completed successfully.")
    quit()
