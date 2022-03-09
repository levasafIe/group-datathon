import re
import statistics
import pandas as pd
from datasets import load_dataset
import pandas as pd
import nltk
from nltk.tokenize.toktok import ToktokTokenizer
from itertools import product

run_conf = {'get_data': False, 'get_final_dataset': False}

if run_conf['get_data']:
    tokenizer = ToktokTokenizer()
    dataset = load_dataset("amazon_us_reviews", 'Mobile_Electronics_v1_00')
    dataset = dataset.data['train']
    table_df = dataset.table
    review_df = table_df.to_pandas()
    review_df['review_date'] = pd.to_datetime(review_df['review_date'])


def get_only_customers_with_more_than_one_review(rev_df):
    rev_df_more_than_one = rev_df.groupby("customer_id").filter(lambda x: x['customer_id'].size > 1)
    return rev_df_more_than_one


def get_reviews_per_user_and_week(rev_df):
    rev_df_more_than_one = get_only_customers_with_more_than_one_review(rev_df)
    rev_df_more_than_one = rev_df_more_than_one[['customer_id', 'review_date']]
    rev_df_more_than_one.sort_values(by=['customer_id', 'review_date'], inplace=True)
    final_df = []
    i = 0
    for customer in rev_df_more_than_one['customer_id'].unique().tolist():
        customer_df = rev_df_more_than_one[rev_df_more_than_one['customer_id'] == customer]
        customer_df_by_week = customer_df.resample('7D', on='review_date').count()
        customer_df_by_week = customer_df_by_week[customer_df_by_week['customer_id'] > 0]
        final_df.append({'customer_id': customer, 'mean_reviews_per_week': customer_df_by_week['customer_id'].mean()})
        i += 1
        if i % 10 == 0:
            print(f'Finished {i} out of {len(rev_df_more_than_one["customer_id"].unique().tolist())}')
    return pd.DataFrame(final_df)
    # prev_user = df.at[0, 'customer_id']
    # prev_review_date = df.at[0, 'review_date']


def remove_stopwords(string, is_lower_case=False):
    stopword_list = nltk.corpus.stopwords.words('english')
    stopword_list.remove('no')
    stopword_list.remove('not')
    tokens = tokenizer.tokenize(string)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in stopword_list]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]
    filtered_string = ' '.join(filtered_tokens)

    return filtered_string


def remove_special(string, remove_digits=False):
    pattern = r'[^a-zA-z0-9\s]' if not remove_digits else r'[^a-zA-z\s]'
    string = re.sub(pattern, '', string)
    return string


def remove_special_and_stop_words(str_input):
    str_input = str_input.lower()
    str_input = remove_stopwords(str_input)
    str_input = remove_special(str_input)
    return str_input


def jaccard_similarity(list1, list2):
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(set(list1)) + len(set(list2))) - intersection
    return float(intersection) / union


def get_similarity(str1, str2):
    if str1 == str2:
        return True
    str1 = remove_special_and_stop_words(str1)
    str2 = remove_special_and_stop_words(str2)
    str1_list = str1.split()
    str2_list = str2.split()
    if (len(str1_list) > 0) | (len(str2_list) > 0):
        return jaccard_similarity(str1_list, str2_list)
    return -1


def get_average_similarity_of_user_text(list_of_text):
    origin_list = list_of_text.copy()
    list_of_text = list(map(lambda x: remove_special_and_stop_words(x), list_of_text))
    similarities = []
    for txt in list_of_text:
        list_of_text.remove(txt)
        if len(list_of_text) > 0:
            combinations = list(product([txt], list_of_text))
            for c in combinations:
                similarities.append(get_similarity(c[0], c[1]))
    similarities = list(filter(lambda x: x >= 0, similarities))
    return statistics.mean(similarities)


if run_conf['get_data']:
    review_per_user_and_week = get_reviews_per_user_and_week(review_df)
    total_review_per_user = pd.DataFrame(review_df.groupby('customer_id')['review_id'].count()).reset_index()
    total_review_per_user.columns = ['customer_id', 'total_reviews']
    review_df = pd.merge(review_df, review_per_user_and_week, how='left', right_on='customer_id', left_on='customer_id')
    review_df = pd.merge(review_df, total_review_per_user, how='inner', right_on='customer_id', left_on='customer_id')
    review_df['mean_reviews_per_week'] = review_df['mean_reviews_per_week'].fillna(value=1)
    review_df.to_csv('prepared_data_set_for_content_distance.csv')
else:
    review_df = pd.read_csv('../prepared_data_set_for_content_distance.csv')


def get_similarity_average_per_list_of_users(list_of_users):
    counter = 1
    final_ = []
    for user in list_of_users:
        review_df_temp = review_df[review_df['customer_id'] == user]
        list_of_desc = review_df_temp['review_body'].tolist()
        avg_similarity = get_average_similarity_of_user_text(list_of_desc)
        final_.append({'customer_id': user, 'avg_sim': avg_similarity})
        counter += 1
        if counter % 5 == 0:
            print(f'finished {counter} out of {len(list_of_users)}')
    return pd.DataFrame(final_)


def contains_email(string):
    if type(string) != str:
        return 0
    match = re.search(r"\S*@\S*\s?", string)
    if match:
        return 1
    else:
        return 0


def contains_link(string):
    if type(string) != str:
        return 0
    match1 = re.search(r"\S*https?:\S*", string)
    match2 = re.search(r"\S*www\S*", string)
    match3 = re.search(r"\S*.com", string)
    match4 = re.search(r"\S*.net", string)
    if match1 or match2 or match3 or match4:
        return 1
    else:
        return 0


review_df['contains_email'] = review_df['review_body'].apply(lambda x: contains_email(x))
review_df['contains_link'] = review_df['review_body'].apply(lambda x: contains_link(x))

if run_conf['get_final_dataset']:
    more_than_one_review_df = review_df[review_df['total_reviews'] > 1]
    more_than_one_review_df_users = more_than_one_review_df['customer_id'].unique().tolist()
    similarity_per_user = get_similarity_average_per_list_of_users(more_than_one_review_df_users)
    review_df = pd.merge(review_df, similarity_per_user, how='left', left_on='customer_id', right_on='customer_id')
    review_df.to_csv('prepared_data_set_with_sim_metrics.csv')
else:
    review_df = pd.read_csv('../prepared_data_set_with_sim_metrics.csv', index_col=0)
review_df = review_df[review_df['total_reviews'] > 1]
review_df.dropna(subset=['review_body'], inplace=True)
review_df['contains_email'] = review_df['review_body'].apply(lambda x: contains_email(x))
review_df['contains_link'] = review_df['review_body'].apply(lambda x: contains_link(x))
user_feature_of_review_df = review_df[['customer_id', 'mean_reviews_per_week', 'total_reviews', 'avg_sim']]
user_feature_of_review_df.drop_duplicates(inplace=True)
user_df = pd.DataFrame(review_df.groupby('customer_id').agg(unique_products=('product_id', 'nunique'),
                                                            unique_product_parents=('product_parent', 'nunique'),
                                                            mean_start_rating=('star_rating', 'mean'),
                                                            number_of_links_in_comment=('contains_link', 'sum'),
                                                            number_of_email_in_comments=('contains_email', 'sum'),
                                                            number_of_vines=('vine', 'sum'),
                                                            number_of_verified_purchase=('verified_purchase', 'sum')
                                                            )).reset_index()
user_df = pd.merge(user_feature_of_review_df, user_df, how='inner', left_on='customer_id', right_on='customer_id')
user_df['percentage_of_links'] = round(user_df['number_of_links_in_comment'] / user_df['total_reviews'], 2)
user_df['percentage_of_emails'] = round(user_df['number_of_email_in_comments'] / user_df['total_reviews'], 2)
user_df['percentage_of_reviews_from_vine'] = round(user_df['number_of_vines'] / user_df['total_reviews'], 2)
user_df['percentage_of_reviews_verified'] = round(user_df['number_of_verified_purchase'] / user_df['total_reviews'], 2)
user_df.drop(columns=['number_of_links_in_comment', 'number_of_email_in_comments', 'number_of_vines',
                      'number_of_verified_purchase'], inplace=True)
