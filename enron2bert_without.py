import pandas
import numpy
from sentence_transformers import SentenceTransformer
import string
import nltk
from nltk.tokenize import word_tokenize

nltk.download("punkt", download_dir="/Users/jk/Desktop/workspaces/pythonWorkspace/magisterka/ai/nltk_data")

stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 
             'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 
             'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 
             "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 
             'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 
             'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 
             'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 
             'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 
             'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 
             'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 
             'through', 'during', 'before', 'after', 'above', 'below', 'to', 
             'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 
             'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 
             'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 
             'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 
             'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 
             "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 
             'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 
             'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 
             'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', 
             "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 
             'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]

punctuation = string.punctuation

def prepare_columns(length):
    return ["BV{}".format(idx) for idx in range(length-1)] + ["Class"]

def remove_unnecesary_words(sentence):
    tokenized_sentence = word_tokenize(sentence)
    return " ".join([word for word in tokenized_sentence if word not in stopwords and word not in punctuation])

model_name = "all-MiniLM-L6-v2"

model = SentenceTransformer(model_name)

enron_csv_df = pandas.read_csv("enron2csv.csv")
print(enron_csv_df.head(n=1)) 

transformed_data = []
for index in enron_csv_df.index:
    print("Transforming ... {}/{}".format(index+1, enron_csv_df.index.stop), end='\r', flush=True)
    row = numpy.concatenate((model.encode(remove_unnecesary_words(enron_csv_df["Content"][index])), numpy.array([enron_csv_df["Class"][index]])), axis=0)
    transformed_data.append(row)

columns = prepare_columns(len(transformed_data[0]))
enron_bert_df = pandas.DataFrame(transformed_data, columns=columns)
print(enron_bert_df.head(n=5))
enron_bert_df.to_csv("enron2bert_{}_without.csv".format(model_name), index=False)
