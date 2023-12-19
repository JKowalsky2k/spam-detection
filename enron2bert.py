import pandas
import numpy
from sentence_transformers import SentenceTransformer

def prepare_columns(length):
    return ["BV{}".format(idx) for idx in range(length-1)] + ["Class"]

model_name = "all-MiniLM-L6-v2"

model = SentenceTransformer(model_name)

enron_csv_df = pandas.read_csv("enron1.csv")
print(enron_csv_df.head(n=1)) 

transformed_data = []
for index in enron_csv_df.index:
    print("Transforming ... {}/{}".format(index+1, enron_csv_df.index.stop), end='\r', flush=True)
    row = numpy.concatenate((model.encode(enron_csv_df["Content"][index]), numpy.array([enron_csv_df["Class"][index]])), axis=0)
    transformed_data.append(row)

columns = prepare_columns(len(transformed_data[0]))
enron_bert_df = pandas.DataFrame(transformed_data, columns=columns)
print(enron_bert_df.head(n=5))
enron_bert_df.to_csv("enron2bert_{}.csv".format(model_name), index=False)
