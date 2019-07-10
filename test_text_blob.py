from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer

blob = TextBlob("It is very bad, I am so sad", analyzer=NaiveBayesAnalyzer())
print(blob.sentiment.p_pos)
