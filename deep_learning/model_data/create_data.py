# import pandas as pd 
# import openai 
# from openai import OpenAI
# import os 
# '''
# This file creates all the training/test data for the seq2seq model to train on by creating a series of potential nl queries that the model may 
# receive as input and their sql equivalents 
# '''

# # Set your OpenAI API key from the environment variable
# openai.api_key = os.getenv("OPENAI_API_KEY")

# # need to parse the prerequisites, corequisites, exclusions and requirements for key descriptors 
# df = pd.read_csv('data/Final_DB.csv')

# all_key_words =[]
# for index, row in df.iterrows(): 
#     print(row)
#     description = row['description']
#     response = openai.chat.completions.create(
#         model="gpt-4",
#         messages=[
#             {
#                 "role": "user",
#                 "content": f"Please extract the key words from the description of this course: {description}",
#             },
#         ],
#     )
#     key_words = response.choices[0].message.content
#     all_key_words.append(key_words)

# df['key_words'] = all_key_words

# df.to_csv('openai_db.csv')
import os
import pandas as pd
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

# Read the CSV file
df = pd.read_csv('data/Final_DB.csv')

all_key_words = []

# Loop through the DataFrame in batches
batch_size = 10  # You can increase the batch size depending on the request limits

for start in range(0, len(df), batch_size):
    batch = df.iloc[start:start+batch_size]
    descriptions = batch['description'].tolist()  # Extract descriptions for the batch

    # Create a list of messages for OpenAI
    messages = [
        {"role": "user", "content": f"Please extract the key words from the description of this course: {description}"}
        for description in descriptions
    ]

    # Send the batch request to OpenAI
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=messages,
    )

    # Parse the response and collect keywords
    for idx, res in enumerate(response.choices):
        key_words = res.message.content
        all_key_words.append(key_words)
        print(f"Extracted keywords for row {start + idx}: {key_words}")  # Print the keywords as you go

    
# Add the keywords to the DataFrame and save to a new CSV
df['key_words'] = all_key_words
df.to_csv('openai_db.csv', index=False)

print("Finished processing all rows.")


#read unique department values 


# need synonyms 

#{sentence prefix: show me, I want to see, list all, what are the,which, find all, i need, get me, what are the etc.... nl permutations}
#courses taught by instructor X 

#{sentence prefix} courses without any prerequisites

#{sentence prefix} {first, second, third, fourth} year courses without any prereqs

#{sentence prefix} {department code or department name or faculty } courses without any prereqs 

#{sentence prefix} {first, second, third, fourth} year {course code or department or faculty } courses without any prereqs 
#{sentence prefix}  {course code or department or faculty } courses in {first, second, third, fourth} year 


#{sentence prefix} {first, second, third, fourth} year {course code or department or faculty } courses 

#{sentence prefix} all the courses offered by the {department name } faculty 


#What courses require {course_code} as a prerequisite? 

#What are the exclusions of {course_code}? 

#Which classes can I not take if I take {course_code}

#Which {year} courses are full year? 

#Which {year} {department} courses are full year? 

#{prefix} {unit_num}-unit {department/ faculty} courses