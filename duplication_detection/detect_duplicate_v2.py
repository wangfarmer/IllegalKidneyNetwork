import pandas as pd
import datetime
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
file_number = 8
# Load the CSV file into a Pandas DataFrame
df = pd.read_csv(f'ki_review_converted_sorted_filter.csv')
#df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%dT%H:%M:%SZ')
#df['Date'] = df['Date'].dt.strftime('%Y/%m/%d')
#df = df.sort_values(by='Date')
#df = pd.read_csv(f'all.csv', encoding='utf-8')
#df = df.dropna(subset=['main_content'])
#df['main_content'].fillna('', inplace=True)

# Define a threshold for similarity
content_threshold = 0.2
#publisher_threshold = 0.35
#content_threshold_with_different_publisher = 0.7
# Define a function to preprocess the content of a row
def preprocess_text(text):
    if text is None or text == "":
        return None
    # Remove punctuation 
    # text = text.translate(str.maketrans('', '', string.punctuation))
    # Convert to lowercase
    if isinstance(text, str):
        text = text.lower()
    # Remove stop words
    # nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    words = str(text).split()
    words = [word for word in words if word not in stop_words]
    # Check if the document contains any valid tokens
    if not words:
        return None
    text = ' '.join(words)
    return text

# Define a function to compare the similarity of two strings
def is_similar(str1, str2):
    corpus = [preprocess_text(str1), preprocess_text(str2)]
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(corpus)
    sim = cosine_similarity(tfidf)[0][1]
    #print(sim)
    return sim

def format_data(date_string):
    format_data=datetime.datetime.strptime(date_string, "%Y-%m-%dT%H:%M:%SZ")
    #format_data=datetime.datetime.strptime(date_string, "%Y/%m/%d")
    return format_data

# Loop through each row in the DataFrame and compare it to all previous rows
to_remove = []
group_index=0
#df['group']= None
for i, row1 in df.iterrows():
    #print(row1)
    print("start to compare " + str(i))
    
    df.at[i,'group'] = group_index
    for j in range(i+1,df.shape[0]):
        row2 = df.iloc[j]
        if abs((format_data(row1['Date'])-format_data(row2['Date'])).days)>7:
            group_index +=1
            break
        else:
            print(str(j) + " rows start to compare with "+ str(i))
            
            #if content_is_similar(row1['main_content'], row2['main_content']) and publisher_is_similar(row1['Source.Name'], row2['Source.Name']):
            if is_similar(row1['main_content'], row2['main_content'])>content_threshold:
                
                print(str(i) + " has been conflict against " + str(j))
                df.at[j,'group'] = group_index
                
                to_remove.append(i)
                print(str(i) + " has been removed")
                
                break
            group_index +=1
#print(to_remove)
'''
with open("remove_0_2_day3_2.txt", 'r') as file:
    for line in file:
        number = int(line.strip())  # Convert the line to an integer
        to_remove.append(number)
# Remove the similar rows from the DataFrame
'''
df.drop(to_remove, inplace=True)

# Save the cleaned data back to the CSV file
df.to_csv(f'unduplicate.csv', index=False,encoding='utf-8')
 
'''
# Load the CSV file into a Pandas DataFrame
df = pd.read_csv('file_split/final_result_1_test.csv')

# Define a threshold for similarity
threshold = 0.55

# Define a function to compare the similarity of two strings
def is_similar(str1, str2):
    corpus = [str1, str2]
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(corpus)
    sim = cosine_similarity(tfidf)[0][1]
    print(sim)
    return sim >= threshold

# Loop through each row in the DataFrame and compare it to all previous rows
to_remove = []
for i, row1 in df.iterrows():
    print("start to compare " + str(i))
    for j in range(i):
        row2 = df.iloc[j]
        if is_similar(row1['main_content'], row2['main_content']):
            to_remove.append(i)
            print(str(i) + " has been conflict against " + str(j))
            print(str(i) + " has been removed")
            break

# Remove the similar rows from the DataFrame
df.drop(to_remove, inplace=True)

# Save the cleaned data back to the CSV file
df.to_csv('file_split/unduplicate_final_result_1.csv', index=False)

'''