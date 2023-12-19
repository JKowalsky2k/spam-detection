import pandas
import glob

data = []

ham_dataset_path = "./data/enron1/ham"
spam_dataset_path = "./data/enron1/spam"

marged_datasets = glob.glob(f"{ham_dataset_path}/*.txt") + glob.glob(f"{spam_dataset_path}/*.txt")

for file_name in marged_datasets:
    with open(file_name, "r", encoding="ISO-8859-1") as file:
        file_content = " ".join([line.strip() for line in file.readlines()])
        file_class = 0 if "ham" in file_name else 1
        data.append([file_content, file_class])

df = pandas.DataFrame(data, columns=["Content", "Class"]) 
print(df.head(n=5))
print(df.shape)
print(df["Class"].value_counts())

df.to_csv('enron1.csv', index=False)