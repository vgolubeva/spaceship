import pandas as pd
from model import open_data, preprocess_data, split_data, load_model_and_predict

data = {
    "HomePlanet": 'Europa',
    "CryoSleep": False,
    "Destination": 'TRAPPIST-1e',
    "Age": 61,
    "VIP": True,
}

user_input_df = pd.DataFrame(data, index=[0])

train_df = open_data()
train_X_df, _ = split_data(train_df)
full_X_df = pd.concat((user_input_df, train_X_df), axis=0)
preprocessed_X_df = preprocess_data(full_X_df, test=False)

print(preprocessed_X_df)
user_X_df = preprocessed_X_df[:1]
print(user_X_df)


prediction, prediction_probas = load_model_and_predict(user_X_df)

print("## Предсказание")
print(prediction)

print("## Вероятность предсказания")
print(prediction_probas)