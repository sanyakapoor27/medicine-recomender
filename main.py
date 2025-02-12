from fastapi import FastAPI, HTTPException
import dask.dataframe as dd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.porter import PorterStemmer
import uvicorn

# Initialize FastAPI app
app = FastAPI()

# Load data
file_path = "Medicine_Details.csv"  # Ensure this file is available in your deployment
# Load only required columns
useful_columns = ['Medicine Name', 'Composition', 'Uses', 'Manufacturer', 
                  'Excellent Review %', 'Average Review %', 'Poor Review %']

meds = dd.read_csv(file_path, usecols=useful_columns).compute()

# Preprocess data
meds = meds[['Medicine Name', 'Composition', 'Uses', 'Manufacturer', 'Excellent Review %', 'Average Review %', 'Poor Review %']]
meds.drop_duplicates(inplace=True)

# Convert Composition and Uses columns to lists
meds['Composition'] = meds['Composition'].apply(lambda x: [i.strip().replace(" ", "") for i in x.split(' + ')])
meds['Uses'] = meds['Uses'].apply(lambda x: x.split())

# Compute Medicine Score
meds['Medicine Score'] = round((meds['Excellent Review %']/100 * 5.0) + (meds['Average Review %']/100 * 3.0) + (meds['Poor Review %']/100 * 1.0), 2)

# Create tags column
meds['tags'] = meds['Composition'] + meds['Uses']
meds['tags'] = meds['tags'].apply(lambda x: ' '.join(x).lower())

# Feature Vectorization
cv = CountVectorizer(max_features=1000, stop_words='english')
vector = cv.fit_transform(meds['tags']).toarray()
similarity = cosine_similarity(vector)

# Utility functions
def recommend_by_name(medicine):
    if medicine not in meds['Medicine Name'].values:
        return []
    
    med_index = meds[meds['Medicine Name'] == medicine].index[0]
    distances = similarity[med_index]
    med_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:10]
    return [{'name': meds.iloc[i[0]]['Medicine Name'], 'score': meds.iloc[i[0]]['Medicine Score']} for i in med_list]

def recommend_by_composition(composition_queries):
    recommendations = [
        {'name': row['Medicine Name'], 'score': row['Medicine Score']}
        for _, row in meds.iterrows()
        if all(query.lower() in ' '.join(row['Composition']).lower() for query in composition_queries)
    ]
    return sorted(recommendations, key=lambda x: x['score'], reverse=True)[:10] or "No matching medicines found."

def recommend_by_use(use_query):
    recommendations = [
        {'name': row['Medicine Name'], 'score': row['Medicine Score']}
        for _, row in meds.iterrows()
        if all(query.lower() in ' '.join(row['Uses']).lower() for query in use_query)
    ]
    return sorted(recommendations, key=lambda x: x['score'], reverse=True)[:10] or "No medicines found for this disease."

# API Endpoints
@app.get("/recommend/name/{medicine}")
def get_recommendation_by_name(medicine: str):
    return recommend_by_name(medicine)

@app.get("/recommend/composition/")
def get_recommendation_by_composition(composition: str):
    composition_queries = composition.split(",")
    return recommend_by_composition(composition_queries)

@app.get("/recommend/use/")
def get_recommendation_by_use(use: str):
    use_query = use.split(",")
    return recommend_by_use(use_query)

# Run the FastAPI app (for local testing)
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)
