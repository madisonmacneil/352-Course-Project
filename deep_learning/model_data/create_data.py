import os
import json
import time
import re
import pandas as pd
import openai
from openai import OpenAI

# OpenAI API client
client = OpenAI(api_key="sk-77372cc3b1094a9ba27a8adb2faa1153", base_url="https://api.deepseek.com")

# File paths
output_file = 'ds_course_data.json'
failed_batches_file = 'failed_batches.json'

# Read dataset
start_index = 120 # Adjust this for new runs
df = pd.read_csv('data/FINAL_DB.csv')
df = df.iloc[start_index:start_index +300].reset_index(drop=True)

# Load existing data if it exists
if os.path.exists(output_file):
    with open(output_file, 'r') as f:
        try:
            all_courses = json.load(f)
        except json.JSONDecodeError:
            print("⚠️ Couldn't load existing output file. Starting fresh.")
            all_courses = {}
else:
    all_courses = {}

failed_batches = []

def extract_json_from_response(text):
    """
    Cleans model output to extract only JSON content.
    Removes markdown formatting like triple backticks.
    """
    if "```json" in text:
        text = text.split("```json", 1)[-1]
    if "```" in text:
        text = text.split("```", 1)[0]
    return text.strip()


# Batching
def batch_data(data, batch_size=10):
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size], i

# API call
def call_openai_api(messages, retries=1):
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
        )
        return response
    except openai.BadRequestError as e:
        print("🚨 OpenAI API error:", e)
    return None

# Processing
for batch, batch_index in batch_data(list(zip(df['description'], df['requirements'], df['course_code']))):
    print(f"📌 Processing batch {batch_index}...")

    content = "Extract structured course information from the following:\n\n"
    for desc, req, code in batch:
        content += f"**Course_Code**: {code}\n\n"
        content += f"**Description**: {desc}\n"
        content += f"**Requirements**: {req}\n\n"

    messages = [{
        "role": "user",
        "content": f"""
            {content}
            Return ONLY raw JSON. Do NOT wrap the JSON in backticks or markdown (like ```json).
            With this structure:
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
                        "min_level": null
                    }},
                    "keywords": ["word1", "word2", "word3", "word4", "word5"]
                }}
            }}
            Return **ONLY** the JSON structure without additional text. NOT a list.
        """
    }]

    response = call_openai_api(messages)
    if response is None or not response.choices or not response.choices[0].message.content.strip():
        print(f"⚠️ Failed batch {batch_index}")
        failed_batches.append(batch_index + start_index)
        continue

    raw_content = response.choices[0].message.content.strip()
    json_content = extract_json_from_response(raw_content)
    print("🔍 Raw response content:")


    try:
        parsed_data = json.loads(json_content)

        new_courses = 0
        for key, value in parsed_data.items():
            all_courses[key] = value
            new_courses += 1
        print(f"✅ Batch {batch_index} processed successfully. ➕ Added {new_courses} new course(s).")
        print(f"Data to write: {all_courses}")

        try:
            with open(output_file, 'w') as f:
                json.dump(all_courses, f, indent=4)
                f.flush() 

        except Exception as e:
            print(f"Error writing data to file: {e}")

    except json.JSONDecodeError as e:
        print(f"⚠️ JSON decoding error in batch {batch_index}: {e}")
        print("🧾 Cleaned content:")
        print(json_content)
        failed_batches.append(batch_index + start_index)

if failed_batches:
    with open(failed_batches_file, 'w') as f:
        json.dump(failed_batches, f, indent=4)
    print(f"❌ Some batches failed. See {failed_batches_file}")

print("✅ All done.")
