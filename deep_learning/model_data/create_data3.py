import os
import json
import time
import re
import pandas as pd
import sqlite3
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- CONFIG ---
API_KEY = "sk-77372cc3b1094a9ba27a8adb2faa1153"
API_BASE = "https://api.deepseek.com"
DB_PATH = 'db_courses.db'
FAILED_LOG = 'failed_batches.log'
CSV_PATH = 'data/FINAL_DB.csv'
START_INDEX = 1560 #need 210-610 done 0-1560
BATCH_SIZE = 10
NUM_ROWS = 400
MAX_THREADS = 5  # Number of threads to use for parallelism

# --- Setup API Client ---
client = OpenAI(api_key=API_KEY, base_url=API_BASE)

# --- Read Dataset ---
df = pd.read_csv(CSV_PATH)
df = df.iloc[START_INDEX:START_INDEX + NUM_ROWS].reset_index(drop=True)

# --- Setup SQLite DB ---
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()



# --- Utils ---
def extract_json_from_response(text):
    """Strip markdown formatting from model output."""
    if "```json" in text:
        text = text.split("```json", 1)[-1]
    if "```" in text:
        text = text.split("```", 1)[0]
    return text.strip()

def batch_data(data, batch_size):
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size], i

def call_openai_api(messages, retries=1):
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
        )
        return response
    except Exception as e:
        print("🚨 OpenAI API error:", e)
    return None

def process_batch(batch, batch_index):
    """Process a single batch of courses, make API call and insert into DB."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

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
                            "BIOL 201": "B",
                            "BIOL 200": null
                        }},
                        "min_level": 2
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
        return batch_index, None

    raw_content = response.choices[0].message.content.strip()
    json_content = extract_json_from_response(raw_content)

    try:
        parsed_data = json.loads(json_content)
        new_courses = 0
        for code, data in parsed_data.items():
            cursor.execute('''
                INSERT OR REPLACE INTO courses (course_code, json_data)
                VALUES (?, ?)
            ''', (code, json.dumps(data)))
            new_courses += 1
        conn.commit()

        print(f"✅ Batch {batch_index} processed. ➕ {new_courses} course(s) saved.")
        return batch_index, new_courses
    except json.JSONDecodeError as e:
        print(f"❌ JSON error in batch {batch_index}: {e}")
        print("🧾 Raw output:")
        print(json_content)
        return batch_index, None

# --- Start Time ---
start_time = time.time()

# --- Processing Loop with ThreadPoolExecutor ---
failed_batches = []
processed_batches = []

with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
    futures = []
    for batch, batch_index in batch_data(list(zip(df['description'], df['requirements'], df['course_code'])), BATCH_SIZE):
        futures.append(executor.submit(process_batch, batch, batch_index))

    for future in as_completed(futures):
        batch_index, result = future.result()
        if result is None:
            failed_batches.append(batch_index + START_INDEX)
        else:
            processed_batches.append(batch_index + START_INDEX)

# --- End Time ---
end_time = time.time()

# --- Final Steps ---
if failed_batches:
    with open(FAILED_LOG, 'w') as f:
        for batch_id in failed_batches:
            f.write(str(batch_id) + '\n')
    print(f"❌ Some batches failed. See {FAILED_LOG} for batch indices.")

# --- Print Time Taken ---
elapsed_time = end_time - start_time
print(f"✅ All done. Total execution time: {elapsed_time:.2f} seconds.")
