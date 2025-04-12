
"""
Create CSVs specifically for the CSP 
Create course_csp.csv which has Course Code, name of course, num of students, prof
create rooms.csv which has room name, capacity 
"""

import pandas as pd
import random

random.seed(1)

# Load the CSV files
classrooms_df = pd.read_csv('data/classrooms.csv')
courses_df = pd.read_csv('data/working_dbs/courses_db.csv')
instructors_df = pd.read_csv('data/working_dbs/filtered_course_instructors.csv')

# Process instructors data
# Convert the courses_taught column from string to list
instructors_df['courses_taught'] = instructors_df['courses_taught'].apply(lambda x: x.strip('[]').replace("'", "").split(', ') if isinstance(x, str) else [])
courses_df['course_code'] = courses_df['course_code'].str.replace(' ', '')

# Create a dictionary mapping course codes to instructors
course_to_instructor = {}
for _, row in instructors_df.iterrows():
    instructor_name = row['name']
    for course in row['courses_taught']:
        course = course.strip()
        if course:  # Ensure the course code is not empty
            course_to_instructor[course] = instructor_name

# Create the course_csp.csv file
course_csp_data = []
for _, row in courses_df.iterrows():
    course_code = row['course_code']
    # Only include courses that have instructors
    if course_code in course_to_instructor:
        course_csp_data.append({
            'course_code': course_code,
            'name': row['course_name'],
            'num_students': random.randint(20, 150),  # give course random number of students from 20 to 150 
            'prof': course_to_instructor[course_code]
        })

course_csp_df = pd.DataFrame(course_csp_data)
print(course_csp_df.head())

course_csp_df.to_csv('csp/course_csp.csv', index=False)

# Create the rooms.csv file
rooms_data = []
for _, row in classrooms_df.iterrows():
    rooms_data.append({
        'room_name': f"{row['Room Number']}",
        'capacity': row['Seats']
    })

rooms_df = pd.DataFrame(rooms_data)
rooms_df.to_csv('csp/rooms.csv', index=False)

# Print summary statistics
print(f"Generated course_csp.csv with {len(course_csp_df)} courses")
print(f"Generated rooms.csv with {len(rooms_df)} rooms")