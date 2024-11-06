# This bertopic reduces and incresaes the number of topics for models that are above a ceratin threshold or under 

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from sentence_transformers import SentenceTransformer
from bertopic.vectorizers import ClassTfidfTransformer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

from umap import UMAP
from hdbscan import HDBSCAN
import pandas as pd

# Load the dataset
data = pd.read_csv('eclipse_platform_updated.csv')
fixed_bugs = data[data['Resolution'] == 'FIXED']
fixed_bugs = fixed_bugs[fixed_bugs['Assignee Real Name'].notna()]
devNum = fixed_bugs['Assignee Real Name'].value_counts()
devNum = devNum[(devNum > 600) & (~devNum.index.str.startswith('Platform')) & (~devNum.index.str.startswith('platform'))]
print(devNum)
print(len(devNum))

def devDocList(name, data):
    docs = data[data['Assignee Real Name'] == name]
    docs = docs[['Title','Description','Product', 'Component', 'Priority', 'Severity']]
    docs['Title'] = docs['Title'].fillna('')
    docs['Description'] = docs['Description'].fillna('')
    docs['Product'] = docs['Product'].fillna('')
    docs['Component'] = docs['Component'].fillna('')
    docs['Priority'] = docs['Priority'].fillna('') # Just added to test
    docs['Severity'] = docs['Severity'].fillna('')  # Just added to test
    docs['Text'] = docs['Product'] + ' ' + docs['Component'] + ' ' + docs['Title']  + ' ' + docs['Description'] + ' ' + docs['Priority'] + ' ' + docs['Severity']
    return docs

# 1. Embedding
sentence_model = SentenceTransformer('all-mpnet-base-v2')

# 2. Dimensionality Reduction
umap_model = UMAP(n_components=1,
                  n_neighbors=15,
                  min_dist=0.0,
                  metric='correlation')

# 3. Clustering 
hdbscan_model = HDBSCAN(min_cluster_size=20,
                        min_samples=1, 
                        metric='euclidean', 
                        cluster_selection_method='leaf',
                        prediction_data=True,
                        allow_single_cluster=True)

# 4. Token
vec = CountVectorizer(stop_words='english',
                      ngram_range=(1,2))

# 5. Weighting schemes
ctif = ClassTfidfTransformer(reduce_frequent_words=True)

# 6. Fine Tune
representation = KeyBERTInspired()

test_documents = []

count = 0

for dev in devNum.index:

    print(f"Training model for {dev}...")
   
    docs_df = devDocList(dev, data)
   
    docTrain_df, docTest_df = train_test_split(docs_df, test_size=0.2, shuffle=False)
   
    docTest_df['Developer'] = dev
    test_documents.append(docTest_df)
   
    docTrain = docTrain_df['Text'].tolist()
   
    # Create and fit BERTopic model
    topic_model = BERTopic(embedding_model=sentence_model,          #1
                           umap_model=umap_model,                   #2
                           hdbscan_model=hdbscan_model,             #3
                           vectorizer_model=vec,                    #4
                           ctfidf_model=ctif,                       #5
                           representation_model=representation)     #6

    topics, probs = topic_model.fit_transform(docTrain)
   
    # Check the number of topics
    topic_info = topic_model.get_topic_info()
    num_topics = len(topic_info) - 1 # Exclude the outlier topic
   
    if num_topics > 10:
        topic_model.reduce_topics(docs=docTrain, nr_topics=8)
    elif num_topics < 1:
        # Adjust HDBSCAN parameters to reduce outliers
        hdbscan_model = HDBSCAN(min_cluster_size=5,
                                min_samples=1, 
                                metric='euclidean', 
                                cluster_selection_method='leaf',
                                prediction_data=True,
                                allow_single_cluster=False)
        
        topic_model = BERTopic(embedding_model=sentence_model,          #1
                               umap_model=umap_model,                   #2
                               hdbscan_model=hdbscan_model,             #3
                               vectorizer_model=vec,                    #4
                               ctfidf_model=ctif,                       #5
                               representation_model=representation)     #6
        
        topics, probs = topic_model.fit_transform(docTrain)
   
    # Print topic information
    print(topic_model.get_topic_info())
    print(topic_model.get_document_info(docTrain))
    print(probs.mean())
   
    #Save the topic model
    topic_model.save(f'{count}_{dev}')
   
    count += 1

print("All models trained and saved.")

# Concatenate all test documents into a single DataFrame
test_documents_df = pd.concat(test_documents, ignore_index=True)

# Save the test documents to a CSV file
test_documents_df.to_csv('test_documents_platform.csv', index=False)
