import pandas as pd
from sklearn.preprocessing import StandardScaler


dataframe = pd.read_csv('Data/pooled_outputs.csv')

dataframe_ = dataframe.drop(columns=['class'])


print(dataframe_.describe())




scaler = StandardScaler()
standardized_data = scaler.fit_transform(dataframe_)

print(standardized_data)

standardized_df = pd.DataFrame(standardized_data, columns=dataframe_.columns)

standardized_df['class'] = dataframe['class'].values

print(standardized_df.describe())


standardized_df.to_csv('Data/standar_datapack.csv', index=False)





