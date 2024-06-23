import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix

app = Flask(__name__)

data = pd.read_csv('D:/_data.csv')

user_item_matrix = pd.pivot_table(data, index='user_id', columns='product_id', values='rating', fill_value=0)

sparse_matrix = csr_matrix(user_item_matrix.values)

cosine_sim = cosine_similarity(sparse_matrix, sparse_matrix)

def get_top_similar_users(user_id, top_n=5):
    sim_scores = list(enumerate(cosine_sim[user_id]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]
    similar_users = [score[0] for score in sim_scores]
    return similar_users

def recommend_products(user_id):
    similar_users = get_top_similar_users(user_id)
    recommended_products = set()
    for user in similar_users:
        products = user_item_matrix.loc[user_item_matrix.index[user], :]
        recommended_products.update(products.idxmax(axis=1).tolist())
    return list(recommended_products)[:5]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    user_id = int(request.form['user_id'])
    recommended_products = recommend_products(user_id)
    return jsonify({'recommendations': recommended_products})

if __name__ == '__main__':
    app.run(debug=True)
