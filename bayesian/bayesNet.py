import pandas as pd
import numpy as np

# Load course enrollment info (course code, number of students, instructor)
courses_df = pd.read_csv('csp/course_csp_limited.csv')
# Filter for the five courses of interest
course_codes = ["APSC111", "APSC171", "APSC182", "CISC101", "HIST105"]
courses_info = courses_df[courses_df['course_code'].isin(course_codes)]
print(courses_info[['course_code','num_students','prof']])

# Load instructor quality info (name, rating, difficulty)
prof_df = pd.read_csv('data/prof_qaulity_info.csv')
prof_df = prof_df.set_index('name')

# Define mapping of course to department (from Final_DB or known department names)
dept_map = {
    "APSC111": "Applied Science",  # first-year Engineering
    "APSC171": "Applied Science",
    "APSC182": "Applied Science",
    "CISC101": "Computing & Information Science",
    "HIST105": "History"
}

# Initialize list for synthetic data records
data_records = []
np.random.seed(42)  # for reproducibility

for course in course_codes:
    # Course attributes from datasets:
    course_row = courses_info[courses_info['course_code']==course].iloc[0]
    class_size = course_row['num_students']             # class enrollment
    instructor = course_row['prof']
    
    # Define earliest time and first day for each course
    earliest_time = {"APSC111": 10.5, "APSC171": 9.5, "APSC182": 8.5,
                     "CISC101": 14.5, "HIST105": 13.5}
    first_day = {"APSC111": 1, "APSC171": 1, "APSC182": 2,
                 "CISC101": 1, "HIST105": 2}  # 1=Mon, 2=Tue, etc.
    
    # Instructor quality and difficulty
    quality = diff = None
    if instructor in prof_df.index:
        quality = float(prof_df.loc[instructor]['rating_val'])   # teaching quality score
        diff = float(prof_df.loc[instructor]['diff_level'])      # difficulty level score
    else:
        quality = 3.0; diff = 3.0  # default if not found

    # Simulate ~50 students for this course
    for _ in range(50):
        # Student attributes
        gpa = np.clip(np.random.normal(3.0, 0.7), 0.0, 4.3)
        
        # Major selection logic
        if course.startswith("APSC"):
            major = "Applied Science"
        elif course == "CISC101":
            major = "Computing & Information Science" if np.random.rand() < 0.5 else np.random.choice(list(dept_map.values()))
        elif course == "HIST105":
            major = "History" if np.random.rand() < 0.3 else np.random.choice(list(dept_map.values()))
        else:
            major = np.random.choice(list(dept_map.values()))
        
        major_related = 1 if dept_map[course] == major else 0
        course_load = np.random.choice([3,4,5,5,5,6])  # bias towards 5
        class_type = 0  # in-person (all have physical rooms)

        # Participation
        if class_size > 80:
            participation = np.random.normal(0.4, 0.1)
        elif class_size > 50:
            participation = np.random.normal(0.6, 0.1)
        else:
            participation = np.random.normal(0.8, 0.1)
        participation = float(np.clip(participation, 0.0, 1.0))

        # Aptitude
        base_apt = gpa/4.3 + (0.1 if major_related else 0.0)
        aptitude = float(np.clip(np.random.normal(base_apt, 0.1), 0.0, 1.0))

        # Score computation
        score = (50
                 + 1  * earliest_time[course]
                 + -2 * first_day[course]
                 + -5 * class_type
                 + -0.05 * class_size
                 + -2 * course_load
                 + 6  * gpa
                 + 3  * major_related
                 + -4 * diff
                 + 2  * quality
                 + 5  * participation
                 + 12 * aptitude)
        score += np.random.normal(0, 5)
        score = np.clip(score, 0, 100)

        # Assign grade
        if score >= 80: grade = 'A'
        elif score >= 70: grade = 'B'
        elif score >= 60: grade = 'C'
        elif score >= 50: grade = 'D'
        else: grade = 'F'

        data_records.append({
            "course": course,
            "course_time": earliest_time[course],
            "course_day": first_day[course],
            "class_size": class_size,
            "course_load": course_load,
            "class_type": class_type,
            "student_gpa": gpa,
            "student_major": major,
            "major_related": major_related,
            "difficulty": diff,
            "quality": quality,
            "participation": participation,
            "aptitude": aptitude,
            "grade": grade
        })

# Convert to DataFrame
data_df = pd.DataFrame(data_records)
print(data_df.head())
print("Grade distribution:", data_df['grade'].value_counts())
