import os

# Set environment variable to disable oneDNN optimizations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import random
from bertopic import BERTopic
import pandas as pd
from sklearn.metrics import classification_report


def findTopKDevs(dev_models, bugReport, k=5, used_devs=None):
    dev_probabilities = {}
    for dev_name, model in dev_models.items():
        if used_devs and dev_name in used_devs:
            continue
        topics, probabilities = model.transform([bugReport])
        if not isinstance(probabilities[0], (list, tuple)):
            max_prob = probabilities[0]
        else:
            max_prob = max(probabilities[0])
        dev_probabilities[dev_name] = max_prob
    
    sorted_devs = sorted(dev_probabilities.items(), key=lambda item: item[1], reverse=True)
    top_k_devs = [dev for dev, _ in sorted_devs[:k]]
    top_k_probs = [prob for _, prob in sorted_devs[:k]]
    
    return top_k_devs, top_k_probs

# In the case of developers having the same value, this function will choose the best dev
def breakTies(devs, probs, bug_opened_date, data):

    zeroCount = 0
    nonZeroCount = 0
    
    for dev, prob in zip(devs, probs):
        if prob == 0:
            zeroCount += 1
        else:
            nonZeroCount += 1

    if zeroCount > nonZeroCount:
        probs = [1.0 for _ in probs]  

    max_prob = max(probs)
    tied_devs = [dev for dev, prob in zip(devs, probs) if prob == max_prob]
    
    if tied_devs:

        closest_dev = min(
            tied_devs,
            key=lambda dev: abs(pd.to_datetime(data[data['Developer'] == dev]['Created_time']) - bug_opened_date).min()
        )

        return closest_dev
    return None

# def is_active_during(dev, bug_opened_date, data):
#     dev_data = data[data['Developer'] == dev]
#     dev_opened_dates = pd.to_datetime(dev_data['Created_time'])
#     dev_changed_dates = pd.to_datetime(dev_data['Resolved_time'])
#     active_during_opened = ((dev_opened_dates <= bug_opened_date) & (dev_opened_dates >= bug_opened_date)).any()
#     return active_during_opened

def is_active_during(dev, bug_opened_date, data):
    dev_data = data[data['Developer'] == dev]
    dev_opened_dates = pd.to_datetime(dev_data['Created_time'])
    return (dev_opened_dates <= bug_opened_date).any()


def has_experience_with_priority_and_severity(dev, bug_priority, bug_severity, data):
    dev_data = data[data['Developer'] == dev]
    has_experience = ((dev_data['Priority'] == bug_priority) & (dev_data['Severity'] == bug_severity)).any()
    return has_experience

# Creating developer list and adding them to a dictonary
data = pd.read_csv('eclipse_platform_updated.csv')
fixed_bugs = data[data['Resolution'] == 'FIXED']
fixed_bugs = fixed_bugs[fixed_bugs['Assignee Real Name'].notna()]
devNum = fixed_bugs['Assignee Real Name'].value_counts()
devNum = devNum[(devNum > 600) & (~devNum.index.str.startswith('Platform')) & (~devNum.index.str.startswith('platform'))]

developer_names = devNum.index.tolist()
model_names = [f'{i}_{name}' for i, name in enumerate(developer_names)]

dev_model_mapping = dict(zip(developer_names, model_names))

# Loading data
data = pd.read_csv('merged_platform.csv')
data['Text'] = data['Product'].fillna('') + ' ' + data['Component'].fillna('') + ' ' + data['Title'].fillna('') + ' ' + data['Description'].fillna('')
data['Text']

# Loading models
docDictionary = {}
for name, model_file in dev_model_mapping.items():
    model = BERTopic.load(model_file)
    docDictionary[name] = model

dev_product_component_mapping = {}
for name in developer_names:
    dev_docs = data[data['Developer'] == name]
    products = dev_docs['Product'].unique()
    components = dev_docs['Component'].unique()
    dev_product_component_mapping[name] = (products, components)

ordered_docDictionary = docDictionary

# Shuffle data
#data = data.sample(frac=1).reset_index(drop=True) # Shuffling data

# Sort data by 'Created_time' in ascending order
data = data.sort_values(by='Created_time').reset_index(drop=True)

# num = 20
# testingDoc = data['Text'][:num]
# true_devs = data['Developer'][:num]

testingDoc = data['Text']
true_devs = data['Developer']
num = len(testingDoc)
print(num)

# Calculate the total number of bugs for each developer
totalBugs = true_devs.value_counts().to_dict()

dev_index_mapping = {name: index for index, name in enumerate(developer_names)}

predicted_labels = []
top_5_correct = 0

if len(testingDoc) > 0:
    true_labels = [dev_index_mapping.get(dev, -1) for dev in true_devs]

    for i in range(num):
        product = data.iloc[i]['Product']
        component = data.iloc[i]['Component']
        priority = data.iloc[i]['Priority']
        severity = data.iloc[i]['Severity']
        bug_opened_date = pd.to_datetime(data.iloc[i]['Created_time'])

        # Filter developers based on activity dates and experience with priority/severity
        relevant_devs = [dev for dev, (products, components) in dev_product_component_mapping.items() 
                         if product in products and component in components 
                         and is_active_during(dev, bug_opened_date, data)
                         and has_experience_with_priority_and_severity(dev, priority, severity, data)]

        filtered_docDictionary = {dev: ordered_docDictionary[dev] for dev in relevant_devs if dev in ordered_docDictionary}

        if filtered_docDictionary:
            top_5_devs, top_5_probs = findTopKDevs(filtered_docDictionary, testingDoc.iloc[i], k=5)
            
            # Select the best available developer from the top_k list, resolving ties if needed
            best_dev = breakTies(top_5_devs, top_5_probs, bug_opened_date, data) if top_5_devs else None
            
            print(f"Document {i}: True Developer - {true_devs.iloc[i]}")
            print(f"Document {i}: Top 5 Developers - {top_5_devs}")
            print(f"Document {i}: Top 5 Probabilities - {top_5_probs}")
            print(f"Document {i}: Best Developer - {best_dev}")
            print()
            
            if best_dev:
                matched_index = dev_index_mapping.get(best_dev, -1)
                if matched_index != -1:
                    predicted_labels.append(matched_index)
                
                if true_labels[i] in [dev_index_mapping.get(dev, -1) for dev in top_5_devs]:
                    top_5_correct += 1

        # If no relevant developers were found, assign the next best available developer
        if not filtered_docDictionary or not best_dev:
            matched_index = dev_index_mapping.get(relevant_devs[0], -1) if relevant_devs else -1
            predicted_labels.append(matched_index)

    # Classification Report
    report = classification_report(true_labels, predicted_labels, target_names=developer_names, zero_division=0)
    print(report)
    #target_names=developer_names,

    # Top 1 and Top 5 accuracy
    top_1_accuracy = sum(1 for true, pred in zip(true_labels, predicted_labels) if true == pred) / num
    top_5_accuracy = top_5_correct / num
    print(f"Top-1 Accuracy: {top_1_accuracy:.2f}")
    print(f"Top-5 Accuracy: {top_5_accuracy:.2f}")

else:
    print("Not enough documents for evaluation.")
