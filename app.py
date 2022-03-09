from flask import Flask, request, Response
import pandas as pd


app = Flask(__name__)

## waiting for him to upload PORT syntax of variable

df = pd.read_csv('user_df_ready.csv')


# Here we need to ask the profesor if he meant print
@app.route("/predict")
def get_user_data():
    args = request.args
    user_id = args['user_id']
    df_user = df[df['customer_id'] == int(user_id)]
    return str(df_user.at[0, 'mean_reviews_per_week'])


app.run()
