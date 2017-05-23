import pandas as pd
import nltk

def gender_features(name):
    name = name.lower()
    return {
        'last': name[-1],
        'last_two': name[-2:],
        'last_three': name[-3:],
        'first': name[0].lower(),
        'first_two': name[:1],
        #'vowel': sum(name.count(letters) for letters in ['a','e','i','o','u'])
    }

df = pd.read_csv("D:\\feltso\data.csv")
featureset = [(gender_features(df.iloc[i,0]),df.iloc[i,1]) for i in range(len(df))]
train = featureset[: int(0.8*len(featureset))]
test = featureset[int(0.8*len(featureset)):]

classifier = nltk.NaiveBayesClassifier.train(train)
print(nltk.classify.accuracy(classifier, test))