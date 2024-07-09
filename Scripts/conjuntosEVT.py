from sklearn.model_selection import train_test_split
import pandas as pd


dataframe = pd.read_csv('Data/pooled_outputs.csv')

train_data, temp_data = train_test_split(dataframe, test_size=0.4, random_state=42)


val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)


train_data.to_csv('train.csv', index=False)
val_data.to_csv('validation.csv', index=False)
test_data.to_csv('test.csv', index=False)