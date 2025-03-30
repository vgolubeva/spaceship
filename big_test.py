import pandas as pd
from model import load_model_and_predict_array, preprocess_data

df = pd.read_csv("data/test.csv")



df = df[["PassengerId", "HomePlanet", "CryoSleep", "Destination", "Age", "VIP"]]
df.dropna(inplace=True)

df1 = df[["PassengerId"]]
df = df[["HomePlanet", "CryoSleep", "Destination", "Age", "VIP"]]


df = preprocess_data(df, test = False)
prediction = load_model_and_predict_array(df)

#print(prediction)
df2 = pd.DataFrame(prediction, columns = ['Transported'])
df1 = df1.reset_index(drop=True)
df3 = pd.concat([df1,df2], axis = 1)
#print(df3)
#df3.to_csv('data/result.csv', index = False)

compareTo = pd.read_csv("data/sample_submission.csv")
#print(compareTo)
#res = compareTo.join(df3, on="PassengerId", lsuffix="_first", rsuffix="_second")
res = pd.concat([df3,compareTo], axis = 1, join = "inner")
count = 0
for row in res.itertuples():
    if row.Transported == row._4:
        count += 1
print(count, len(res))
print(count / len(res))
#print(list(res["Transported"].values()))


''''#user_input_df = pd.DataFrame(data, index=[0])
user_input_df = df

train_df = open_data()
train_X_df, _ = split_data(train_df)
full_X_df = pd.concat((user_input_df, train_X_df), axis=0)
preprocessed_X_df = preprocess_data(full_X_df, test=False)

user_X_df = preprocessed_X_df[:1]
#print(user_X_df)


prediction, prediction_probas = load_model_and_predict(user_X_df)

print("## Предсказание")
print(prediction)

print("## Вероятность предсказания")
print(prediction_probas)'''