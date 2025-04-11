# import pandas as pd 
# import openai 
# from openai import OpenAI
# import os 
# '''
# This file creates all the training/test data for the seq2seq model to train on by creating a series of potential nl queries that the model may 
# receive as input and their sql equivalents 
# ''' 

import os
import json
import time
import re
import pandas as pd
import openai

# Load OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")
if openai.api_key:
    print("✅ OpenAI API key loaded successfully.")
else:
    print("❌ OpenAI API key not found. Make sure it's set correctly.")

# Read dataset
start_index = 400 # Modify this to start processing from a specific row (excluding headers)
df = pd.read_csv('data/FINAL_DB.csv')
df = df.iloc[start_index:].reset_index(drop=True)  # Exclude rows before start_index
df = df.head(600)

# Define batching function
def batch_data(data, batch_size=10):
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size], i

# Initialize result storage
all_courses = {}
failed_batches = []

# Define API request function with retry mechanism
def call_openai_api(messages, retries=1):
    try:
            response = openai.chat.completions.create(
                model="gpt-4",
                messages=messages,
            )
            return response  # Return response if successful
    except openai.BadRequestError as e:
            print(f"🚨 OpenAI API error ")

    return None  # Return None if all attempts fail

# Process dataset in batches
for batch, batch_index in batch_data(list(zip(df['description'], df['requirements'], df['course_code']))):
    print(f"📌 Processing batch {batch_index}...")

    # Construct request content
    content = "Extract structured course information from the following:\n\n"
    for desc, req, code in batch:
        content += f"**Course_Code**: {code}\n\n"
        content += f"**Description**: {desc}\n"
        content += f"**Requirements**: {req}\n\n"

    messages = [{
        "role": "user",
        "content": f"""
            {content}
            Return **only valid JSON output** with these structures:

            ```json
            {{
                "course code": {{
                    "exclusions": {{
                        "complete_exclusions": ["COURSE_101", "COURSE_102"],
                        "excluded_if_taken_after": [],
                        "unit_cap_between": {{"max_units": "max_units_between_course_set", "courses": ["COURSE_201", "COURSE_202"]}}
                    }},
                    "permissions": {{
                        "instructor_permission": false,
                        "other_components": {{
                            "tutorial/seminar": false,
                            "lab": false
                        }}
                    }},
                    "requirements": {{
                        "registered_in_program": null,
                        "gpa_min": null,
                        "prereq_grade_min": {{
                            "BIOL 205": "B",
                            "BIOL 200": null
                        }},
                        "level": null
                    }},
                    "keywords": ["word1", "word2", "word3", "word4", "word5"]
                }}
            }}
            ```
            Return **ONLY** the JSON structure without additional text. NOT a list.
        """
    }]

    # Call OpenAI API with retries
    response = call_openai_api(messages)
    
    if response is None or not response.choices or not response.choices[0].message.content.strip():
        print(f"⚠️ Failed batch {batch_index}, saving index for later processing.")
        failed_batches.append(batch_index)
        continue  # Skip to next batch

    # Parse response JSON
    json_content = response.choices[0].message.content.strip()
    try:
        parsed_data = json.loads(json_content)
        for course_code, course_info in parsed_data.items():
            if course_code not in all_courses:
                all_courses[course_code] = {
                    "exclusions": {},
                    "permissions": {},
                    "requirements": {},
                    "keywords": []
                }
            for section, content in course_info.items():
                all_courses[course_code][section] = content

        print(f"✅ Batch {batch_index} processed successfully.")

    except json.JSONDecodeError:
        print(f"⚠️ JSON decoding error in batch {batch_index}.")
        failed_batches.append(batch_index+start_index)

    # Save results after each batch
    with open('course_data.json', 'a') as f:
        json.dump(all_courses, f, indent=4)

# Save failed batch indices
if failed_batches:
    with open('failed_batches.json', 'w') as f:
        json.dump(failed_batches, f, indent=4)
    print(f"❌ Some batches failed. Saved failed batch indices to failed_batches.json.")

print("✅ Processing complete.")
