# import pandas as pd 
# import openai 
# from openai import OpenAI
# import os 
# '''
# This file creates all the training/test data for the seq2seq model to train on by creating a series of potential nl queries that the model may 
# receive as input and their sql equivalents 
# ''' 

import os
import pandas as pd
import openai
#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
openai.api_key = os.getenv("OPENAI_API_KEY")
if openai.api_key:
    print(openai.api_key)
    print("✅ OpenAI API key loaded successfully.")
else:
    print("❌ OpenAI API key not found. Make sure it's set correctly.")

'''
{
    "complete_exclusions": ["COURSE_101", "COURSE_102"],  # Cannot take at all
    "excluded_if_taken_after": [],  # Cannot take if already completed a later course
    "unit_cap_between": {"max_units": 6, "courses": ["COURSE_201", "COURSE_202"]}
}
'''
'''
{
    "instructor_permission": False,
    "other_components": {
        "tutorial": ,
        "lab": False
    }
}
'''
'''
{
    "registered_in": "SomeProgram",  # Program/degree student must be registered in
    "gpa_min": 2.5,  # Minimum GPA required
    "prereq_grade_min": {  # Dictionary for required course grades
        "BIOL 205": "B",
        "BIOL 200": "C+"
    },
    "dept_permission": True,
    "level": 3,  # Minimum level/year required
    "cumulative_gpa": 3.0  # Overall GPA required
}
'''
import re
import json
import openai
import pandas as pd

# Initialize result dictionary to store course-specific information
all_courses = {}

# Read the entire dataset
df = pd.read_csv('data/FINAL_DB.csv')
df = df.head(1000)

# Process prereq_string to extract prerequisite groupings
def parse_prerequisites(prereq_string):
    # Step 1: Split by "or" to get separate groups
    or_groups = prereq_string.split(" or ")

    # Step 2: For each "or" group, split by "and" to get individual courses
    parsed_groups = []
    for group in or_groups:
        # Find all courses (e.g., "Course 1", "Course 2", etc.)
        courses = re.findall(r'\bCourse \d+\b', group)
        parsed_groups.append(tuple(courses))  # Store courses as a tuple in the group

    return parsed_groups


# Function to split data into manageable batches for API calls
def batch_data(data, batch_size=10):
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]

# Prepare the descriptions and requirements to be included in the message
course_codes = df['course_code'].tolist()  # Assuming course codes are in a column 'course_code'
descriptions = df['description'].tolist()
requirements = df['requirements'].tolist()

# Process the dataset in batches
i = 0
for batch in batch_data(list(zip(descriptions, requirements, course_codes))):
    print('========================', i, '========================')
    # Construct the content of the single message
    content = "Extract structured course information from the following:\n\n"

    for desc, req, code in batch:
        content += f"**Course_Code**: {code}\n\n"
        content += f"**Description**: {desc}\n"
        content += f"**Requirements**: {req}\n\n"

    # Send the message to OpenAI
    messages = [
        {
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
                                "BIOL 200": null, 
                            }},
                            "prereq_string": "(Course 1 and Course 2) or Course 3",  # All prerequisite courses
                            "mandatory_dept_permission": false, 
                            "level": null
                            "needed_faculty_units": {{"faculty": needed_units}} 
                        }},
                        "keywords": ["word1", "word2", "word3", "word4", "word5"]
                    }}
                }}
                ```

                Return **only** the JSON structure without additional text. 
            """
        }
    ]

    # Send the message to the OpenAI API
    try:
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=messages,
        )
        
        # Check if response contains valid content
        if not response.choices or not response.choices[0].message.content.strip():
            print("⚠️ Empty response from OpenAI, skipping batch.")
            continue

        json_content = response.choices[0].message.content.strip()

        try:
            parsed_data = json.loads(json_content)
        except json.JSONDecodeError:
            print(f"⚠️ Error decoding JSON: {json_content}")
            continue  # Skip to next batch

        # Process the response
        for idx, res in enumerate(response.choices):
            parsed_data = json.loads(res.message.content.strip())  # Ensure valid JSON
            print(parsed_data)
            try:

                for course_code, course_info in parsed_data.items():
                    try:
                        print(f"Course Code: {course_code}")

                        if course_code not in all_courses:
                            all_courses[course_code] = {
                                "exclusions": {},
                                "permissions": {},
                                "requirements": {},
                                "keywords": []
                            }

           
                        for section, content in course_info.items():
                            print(f"  {section.capitalize()}: {content}")
                            all_courses[course_code][section] = content

                        print()

                    except KeyError as e:
                        print(f"Key error encountered in course {course_code}: {e}")
                    except Exception as e:
                        print(f"Skipped a course {course_code} due to error: {e}")

            except json.JSONDecodeError:
                print("Error decoding JSON in response.")

    except openai.BadRequestError as e:
        print(f"🚨 OpenAI API error: {e}")

    with open('openai_course_data.json', 'w') as f:
            json.dump(all_courses, f, indent=4)

    print(f"✅ Finished Batch: {i}, data saved.")

    i+=1

# Once processing is complete, save the result to a CSV or JSON file
# For example, to save it as a JSON file


print("✅ Finished processing.")
