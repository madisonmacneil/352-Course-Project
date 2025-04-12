# This script parses a class schedule, identifies the student's major, collects course and instructor info,
# prompts for student GPA and prerequisite grades, trains a Bayesian Network (Naive Bayes) model on synthetic data,
# and outputs the grade probability distribution for each course along with the model's accuracy.

# Section 1: Parse weekly class schedule and identify enrolled courses and the student's major
from collections import Counter

try:
    with open("csp/hypothetical_schedule.txt", "r") as f:
        schedule_lines = f.readlines()
except FileNotFoundError:
    print("Error: 'hypothetical_schedule.txt' not found. Ensure the schedule file is available.")
    exit(1)

courses_list = []
course_name_map = {}
instructor_map = {}

for line in schedule_lines:
    line = line.strip()
    if not line or line.endswith(":") or line.startswith("Room:"):
        continue  # skip blank lines, day headings, and room lines
    if " - " in line:
        # Example line format: "09:30 - APSC171: Calculus I (Kexue Zhang)"
        parts = line.split(" - ", 1)
        if len(parts) < 2:
            continue
        course_part = parts[1]  # e.g., "APSC171: Calculus I (Kexue Zhang)"
        if ":" in course_part:
            code, rest = course_part.split(":", 1)
            code = code.strip()
            course_name = rest.split("(")[0].strip()
            instr = ""
            if "(" in rest:
                instr = rest.split("(")[1].replace(")", "").strip()
            # Store the course code, name, and instructor
            courses_list.append(code)
            course_name_map[code] = course_name
            instructor_map[code] = instr

# Determine unique enrolled courses and their frequencies
course_counts = Counter(courses_list)
enrolled_courses = list(course_counts.keys())

# Identify the course that appears three times (the primary course, assumed to be in the major field)
major_course = None
for code, count in course_counts.items():
    if count == 3:
        major_course = code
        break
if major_course is None:
    # If no course appears exactly 3 times, take the one with the highest frequency
    major_course = max(course_counts, key=course_counts.get)

# Define the student's major as the subject prefix of that course code (letters part of the code)
major_prefix = "".join([c for c in major_course if not c.isdigit()])
student_major_prefix = major_prefix

print(f"Identified courses: {enrolled_courses}")
print(f"The course '{major_course}' appears {course_counts[major_course]} times. "
      f"Assuming major in '{student_major_prefix}' department.")

# Section 2: Gather course attributes (class size) and instructor info (professor ratings, difficulty)
import pandas as pd

try:
    courses_df = pd.read_csv("csp/course_csp_limited.csv")
    prof_df = pd.read_csv("data/prof_qaulity_info.csv")
except FileNotFoundError as e:
    print(f"Error: Data file not found ({e.filename}). Please ensure all required CSV files are present.")
    exit(1)

# Create lookup for class sizes (course_code -> number of students)
class_size_map = dict(zip(courses_df['course_code'], courses_df['num_students']))
# Create lookup for professor quality info (name -> (rating_val, diff_level))
prof_info_map = {}
for _, row in prof_df.iterrows():
    name = row['name']
    rating = row['rating_val']
    diff = row['diff_level']
    # Convert rating and difficulty to float, or use default if not available
    try:
        rating_val = float(rating)
    except:
        rating_val = 3.0  # default medium rating
    try:
        diff_val = float(diff)
    except:
        diff_val = 3.0    # default medium difficulty
    prof_info_map[name] = (rating_val, diff_val)

# Retrieve relevant info for each enrolled course
course_info = {}
for code in enrolled_courses:
    # Get class size
    class_size = class_size_map.get(code)
    # Get instructor name and then professor rating/difficulty
    instr_name = instructor_map.get(code, "")
    rating_val, diff_val = None, None
    if instr_name in prof_info_map:
        rating_val, diff_val = prof_info_map[instr_name]
    course_info[code] = {
        "name": course_name_map.get(code, ""),
        "instructor": instr_name,
        "class_size": class_size,
        "prof_rating": rating_val,
        "prof_difficulty": diff_val
    }

# Section 3: Prompt user for current GPA

print("\nLet's learn more about you!")
morning_pref = None
while morning_pref not in ["morning", "night"]:
    morning_pref = input("Are you a morning person or a night owl? (Enter 'morning' or 'night'): ").lower()

# Collect info for each course
friend_map = {}
for course in enrolled_courses:
    response = ""
    while response not in ["yes", "no"]:
        response = input(f"Do you have friends in {course}? (yes/no): ").lower()
    friend_map[course] = response == "yes"

print("\n\U0001F4D6 Your Schedule for Next Semester:")
print(f"You are a {student_major_prefix} Major\n")

# Print detailed course info
for code in enrolled_courses:
    info = course_info.get(code, {})
    instructor = info.get("instructor", "N/A")
    quality = info.get("prof_rating", "N/A")
    difficulty = info.get("prof_difficulty", "N/A")
    class_size = info.get("class_size", "Unknown")
    course_name = info.get("name", "")
    related = "Yes" if code.startswith(student_major_prefix) else "No"

    # Try to get time and day from schedule
    course_times = []
    course_days = []
    for line in schedule_lines:
        if code in line:
            time_str = line.split('-')[0].strip()
            if time_str:
                course_times.append(time_str)
        elif any(day in line for day in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]):
            current_day = line.replace(":", "")
            if code in schedule_lines[schedule_lines.index(line)+1]:
                course_days.append(current_day)

    print(f"\U0001F4D8 {code} - {course_name}")
    print(f"\U0001F468‍\U0001F3EB Instructor: {instructor}")
    print(f"⭐ Rating: {quality} | 💣 Difficulty: {difficulty}")
    print(f"🕘 Time(s): {', '.join(course_times) if course_times else 'N/A'}")
    print(f"📅 Days: {', '.join(course_days) if course_days else 'N/A'}")
    print(f"🏫 Class Size: {class_size} students")
    print(f"🧠 Related to Your Major: {related}")
    print("".ljust(40, "-"))

user_gpa = None
while True:
    try:
        user_gpa = float(input("Enter your current GPA (0.0 - 4.3): "))
    except ValueError:
        print("Invalid input. Please enter a numeric GPA (e.g., 3.5).")
        continue
    if 0.0 <= user_gpa <= 4.3:
        break
    else:
        print("Please enter a GPA between 0.0 and 4.3.")

# Section 4: Check for prerequisites for upper-year courses and prompt for grades if needed
try:
    prereq_df = pd.read_csv("bayesian/complete_courses.csv", usecols=['course_code', 'prereq_codes'])
except FileNotFoundError:
    print("Warning: 'complete_courses.csv' not found. Skipping prerequisite checks.")
    prereq_df = pd.DataFrame(columns=['course_code', 'prereq_codes'])

# Build a dictionary for prerequisites (e.g., "CISC200" -> ["CISC101"])
prereq_lookup = {}
for _, row in prereq_df.iterrows():
    code = str(row['course_code'])            # e.g., "CISC 200"
    code_no_space = code.replace(" ", "")     # e.g., "CISC200"
    prereq_list_str = row['prereq_codes']
    if isinstance(prereq_list_str, str) and prereq_list_str.strip():
        prereq_list = [c.strip() for c in prereq_list_str.split(',')]
    else:
        prereq_list = []
    prereq_lookup[code_no_space] = prereq_list

# Determine prerequisite average grade category for each course (if applicable)
prereq_avg_cat = {code: 'None' for code in enrolled_courses}  # default to 'None'
for code in enrolled_courses:
    # Only consider prerequisites for courses that are not 100-level (course number >= 200)
    number_part = "".join([c for c in code if c.isdigit()]) or "0"
    course_num = int(number_part)
    if course_num >= 200:
        prereqs = prereq_lookup.get(code, [])
        if prereqs:
            prereq_grades = []
            for prereq_code in prereqs:
                prompt = f"Enter your grade (0-100) in prerequisite course {prereq_code}: "
                while True:
                    try:
                        grade_val = int(input(prompt))
                    except ValueError:
                        print("Invalid input. Please enter an integer grade from 0 to 100.")
                        continue
                    if 0 <= grade_val <= 100:
                        prereq_grades.append(grade_val)
                        break
                    else:
                        print("Grade should be between 0 and 100.")
            # Calculate average prerequisite grade and categorize it
            if prereq_grades:
                avg_grade = sum(prereq_grades) / len(prereq_grades)
                if avg_grade >= 80:
                    prereq_avg_cat[code] = 'High'
                elif avg_grade >= 60:
                    prereq_avg_cat[code] = 'Medium'
                else:
                    prereq_avg_cat[code] = 'Low'
        else:
            prereq_avg_cat[code] = 'None'

# Section 5: Generate synthetic data and train a Bayesian Network (Naive Bayes) model
import numpy as np
from sklearn.naive_bayes import CategoricalNB
from sklearn.model_selection import train_test_split

# Define possible categories for each feature and the target grade
difficulty_levels   = ['Low', 'Medium', 'High']
quality_levels      = ['Low', 'Medium', 'High']
class_size_levels   = ['Small', 'Medium', 'Large']
timing_levels       = ['Morning', 'Afternoon', 'Evening']
gpa_levels          = ['Low', 'Medium', 'High']
major_categories    = ['Engineering', 'Computing', 'History', 'Biology', 'Other']
match_categories    = ['No', 'Yes']
participation_levels= ['Low', 'Medium', 'High']
prereq_avg_levels   = ['None', 'Low', 'Medium', 'High']
grade_levels        = ['A', 'B', 'C', 'D', 'F']

# Set up conditional probability distributions P(feature | Grade) for synthetic data
# (These are assumptions to simulate a realistic scenario)
prob_dist = {
    'difficulty': {
        'A': {'Low': 0.5, 'Medium': 0.4, 'High': 0.1},
        'B': {'Low': 0.4, 'Medium': 0.4, 'High': 0.2},
        'C': {'Low': 0.3, 'Medium': 0.5, 'High': 0.2},
        'D': {'Low': 0.2, 'Medium': 0.4, 'High': 0.4},
        'F': {'Low': 0.1, 'Medium': 0.3, 'High': 0.6}
    },
    'quality': {
        'A': {'Low': 0.1, 'Medium': 0.3, 'High': 0.6},
        'B': {'Low': 0.2, 'Medium': 0.4, 'High': 0.4},
        'C': {'Low': 0.3, 'Medium': 0.5, 'High': 0.2},
        'D': {'Low': 0.4, 'Medium': 0.4, 'High': 0.2},
        'F': {'Low': 0.6, 'Medium': 0.3, 'High': 0.1}
    },
    'class_size': {
        'A': {'Small': 0.5, 'Medium': 0.3, 'Large': 0.2},
        'B': {'Small': 0.4, 'Medium': 0.4, 'Large': 0.2},
        'C': {'Small': 0.3, 'Medium': 0.5, 'Large': 0.2},
        'D': {'Small': 0.2, 'Medium': 0.5, 'Large': 0.3},
        'F': {'Small': 0.1, 'Medium': 0.4, 'Large': 0.5}
    },
    'timing': {
        'A': {'Morning': 0.2, 'Afternoon': 0.6, 'Evening': 0.2},
        'B': {'Morning': 0.3, 'Afternoon': 0.5, 'Evening': 0.2},
        'C': {'Morning': 0.4, 'Afternoon': 0.4, 'Evening': 0.2},
        'D': {'Morning': 0.5, 'Afternoon': 0.4, 'Evening': 0.1},
        'F': {'Morning': 0.6, 'Afternoon': 0.3, 'Evening': 0.1}
    },
    'gpa': {
        'A': {'Low': 0.1, 'Medium': 0.2, 'High': 0.7},
        'B': {'Low': 0.1, 'Medium': 0.5, 'High': 0.4},
        'C': {'Low': 0.2, 'Medium': 0.6, 'High': 0.2},
        'D': {'Low': 0.5, 'Medium': 0.4, 'High': 0.1},
        'F': {'Low': 0.7, 'Medium': 0.3, 'High': 0.0}
    },
    'major': {
        # Assume no strong correlation between major and grade for synthetic data
        'A': {'Engineering': 0.25, 'Computing': 0.25, 'History': 0.25, 'Biology': 0.25, 'Other': 0.0},
        'B': {'Engineering': 0.25, 'Computing': 0.25, 'History': 0.25, 'Biology': 0.25, 'Other': 0.0},
        'C': {'Engineering': 0.25, 'Computing': 0.25, 'History': 0.25, 'Biology': 0.25, 'Other': 0.0},
        'D': {'Engineering': 0.25, 'Computing': 0.25, 'History': 0.25, 'Biology': 0.25, 'Other': 0.0},
        'F': {'Engineering': 0.25, 'Computing': 0.25, 'History': 0.25, 'Biology': 0.25, 'Other': 0.0}
    },
    'major_match': {
        'A': {'No': 0.4, 'Yes': 0.6},
        'B': {'No': 0.5, 'Yes': 0.5},
        'C': {'No': 0.6, 'Yes': 0.4},
        'D': {'No': 0.6, 'Yes': 0.4},
        'F': {'No': 0.7, 'Yes': 0.3}
    },
    'participation': {
        'A': {'Low': 0.1, 'Medium': 0.3, 'High': 0.6},
        'B': {'Low': 0.2, 'Medium': 0.5, 'High': 0.3},
        'C': {'Low': 0.3, 'Medium': 0.6, 'High': 0.1},
        'D': {'Low': 0.5, 'Medium': 0.4, 'High': 0.1},
        'F': {'Low': 0.7, 'Medium': 0.3, 'High': 0.0}
    },
    'prereq_avg': {
        'A': {'None': 0.4, 'Low': 0.05, 'Medium': 0.15, 'High': 0.4},
        'B': {'None': 0.5, 'Low': 0.1, 'Medium': 0.2, 'High': 0.2},
        'C': {'None': 0.5, 'Low': 0.2, 'Medium': 0.2, 'High': 0.1},
        'D': {'None': 0.5, 'Low': 0.3, 'Medium': 0.15, 'High': 0.05},
        'F': {'None': 0.6, 'Low': 0.25, 'Medium': 0.1, 'High': 0.05}
    }
}
# Prior probabilities for each grade (assuming more B/C grades in general)
grade_prior = {'A': 0.20, 'B': 0.30, 'C': 0.25, 'D': 0.15, 'F': 0.10}

# Generate synthetic dataset
N = 2000  # number of synthetic student-course records
synthetic_data = []
np.random.seed(0)  # for reproducibility
grades = list(grade_prior.keys())
grade_probs = list(grade_prior.values())

for _ in range(N):
    # Sample a grade for the synthetic record
    grade = np.random.choice(grades, p=grade_probs)
    # Sample features based on the conditional distributions for this grade
    diff_val = np.random.choice(difficulty_levels, p=list(prob_dist['difficulty'][grade].values()))
    qual_val = np.random.choice(quality_levels,    p=list(prob_dist['quality'][grade].values()))
    class_val = np.random.choice(class_size_levels, p=list(prob_dist['class_size'][grade].values()))
    time_val  = np.random.choice(timing_levels,      p=list(prob_dist['timing'][grade].values()))
    gpa_val   = np.random.choice(gpa_levels,         p=list(prob_dist['gpa'][grade].values()))
    major_val = np.random.choice(major_categories,   p=list(prob_dist['major'][grade].values()))
    match_val = np.random.choice(match_categories,   p=list(prob_dist['major_match'][grade].values()))
    part_val  = np.random.choice(participation_levels, p=list(prob_dist['participation'][grade].values()))
    prereq_val= np.random.choice(prereq_avg_levels,    p=list(prob_dist['prereq_avg'][grade].values()))
    synthetic_data.append([diff_val, qual_val, class_val, time_val, gpa_val,
                           major_val, match_val, part_val, prereq_val, grade])

# Map categories to numeric codes for model training
difficulty_map   = {lvl: idx for idx, lvl in enumerate(difficulty_levels)}
quality_map      = {lvl: idx for idx, lvl in enumerate(quality_levels)}
class_map        = {lvl: idx for idx, lvl in enumerate(class_size_levels)}
timing_map       = {lvl: idx for idx, lvl in enumerate(timing_levels)}
gpa_map          = {lvl: idx for idx, lvl in enumerate(gpa_levels)}
major_map        = {cat: idx for idx, cat in enumerate(major_categories)}
match_map        = {cat: idx for idx, cat in enumerate(match_categories)}
participation_map= {lvl: idx for idx, lvl in enumerate(participation_levels)}
prereq_map       = {lvl: idx for idx, lvl in enumerate(prereq_avg_levels)}
grade_map        = {gr: idx for idx, gr in enumerate(grade_levels)}

# Prepare feature matrix X and label vector y for the synthetic dataset
X = []
y = []
for record in synthetic_data:
    diff_val, qual_val, class_val, time_val, gpa_val, major_val, match_val, part_val, prereq_val, grade = record
    X.append([
        difficulty_map[diff_val],
        quality_map[qual_val],
        class_map[class_val],
        timing_map[time_val],
        gpa_map[gpa_val],
        major_map[major_val],
        match_map[match_val],
        participation_map[part_val],
        prereq_map[prereq_val]
    ])
    y.append(grade_map[grade])
X = np.array(X)
y = np.array(y)

# Split synthetic data into training and testing sets for model evaluation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Train a Categorical Naive Bayes model (a simple Bayesian Network) on the training data
model = CategoricalNB()
model.fit(X_train, y_train)

# Section 6: Predict grade probability distribution for each of the student's courses
# Prepare feature inputs for each actual enrolled course
user_X = []
for code in enrolled_courses:
    info = course_info.get(code, {})
    # Professor difficulty category (thresholds: <=2.5 Low, >=3.6 High, otherwise Medium)
    diff_val = info.get("prof_difficulty")
    if diff_val is None:
        diff_cat = 'Medium'
    else:
        diff_cat = 'Low' if diff_val <= 2.5 else ('High' if diff_val >= 3.6 else 'Medium')
    # Professor quality (rating) category (thresholds: <3.0 Low, >=4.0 High, otherwise Medium)
    rating_val = info.get("prof_rating")
    if rating_val is None:
        qual_cat = 'Medium'
    else:
        qual_cat = 'Low' if rating_val < 3.0 else ('High' if rating_val >= 4.0 else 'Medium')
    # Class size category (<=50 Small, >100 Large, otherwise Medium)
    size_val = info.get("class_size")
    if size_val is None:
        class_cat = 'Medium'
    else:
        class_cat = 'Small' if size_val <= 50 else ('Large' if size_val > 100 else 'Medium')
    # Timing category (determine if class is mostly Morning, Afternoon, or Evening)
    timing_cat = 'Morning'
    hours = []
    for line in schedule_lines:
        if code in line:
            time_str = line.split('-')[0].strip()  # e.g., "09:30"
            if time_str:
                try:
                    hour = int(time_str.split(':')[0])
                except:
                    hour = 0
                hours.append(hour)
    if hours:
        morning_count = sum(1 for h in hours if h < 12)
        afternoon_count = sum(1 for h in hours if 12 <= h < 17)
        evening_count = sum(1 for h in hours if h >= 17)
        if evening_count > max(morning_count, afternoon_count):
            timing_cat = 'Evening'
        elif afternoon_count > morning_count:
            timing_cat = 'Afternoon'
        else:
            timing_cat = 'Morning'
    
     # morning pref
    if morning_pref == "morning" and timing_cat == 'Morning':
        gpa_adj = 0.2
    elif morning_pref == "night" and timing_cat == 'Evening':
        gpa_adj = 0.2
    else:
        gpa_adj = 0.0
    # GPA category (user's GPA: <2.5 Low, 2.5-3.49 Medium, >=3.5 High)
    adjusted_gpa = min(4.3, user_gpa + gpa_adj)
    if adjusted_gpa >= 3.5:
        gpa_cat = 'High'
    elif adjusted_gpa >= 2.5:
        gpa_cat = 'Medium'
    else:
        gpa_cat = 'Low'
    # Major category (map the student's major prefix to one of the defined major categories)
    if student_major_prefix in ['APSC', 'MECH', 'ELEC', 'CIVL', 'CHEE', 'MINE', 'ENPH', 'MTHE', 'ENGR', 'ENSC']:
        major_cat = 'Engineering'
    elif student_major_prefix in ['CISC', 'SOFT', 'CMPE']:
        major_cat = 'Computing'
    elif student_major_prefix == 'HIST':
        major_cat = 'History'
    elif student_major_prefix == 'BIOL':
        major_cat = 'Biology'
    else:
        major_cat = 'Other'

    # Major-course match (Yes if course prefix matches major prefix, else No)
    course_prefix = "".join([c for c in code if not c.isdigit()])
    match_cat = 'Yes' if course_prefix == student_major_prefix else 'No'
    # Participation category (assume Medium participation for all courses)
    if friend_map[code]:
        part_cat = 'High'
    else:
        part_cat = 'Medium'
    # Prerequisite average grade category from earlier input
    prereq_cat = prereq_avg_cat.get(code, 'None')
    # Append the encoded feature vector for this course
    user_X.append([
        difficulty_map[diff_cat],
        quality_map[qual_cat],
        class_map[class_cat],
        timing_map[timing_cat],
        gpa_map[gpa_cat],
        major_map.get(major_cat, major_map['Other']),
        match_map[match_cat],
        participation_map[part_cat],
        prereq_map[prereq_cat]
    ])
user_X = np.array(user_X)

# Get predicted probability distribution for grades (A-F) for each course
proba_distributions = model.predict_proba(user_X)

print("\nPredicted Grade Distributions:")
# Print header for table
header = f"{'Course':8}{'A':6}{'B':6}{'C':6}{'D':6}{'F':6}"
print(header)
# Print each course with probabilities for A, B, C, D, F
for code, probs in zip(enrolled_courses, proba_distributions):
    # The model classes_ are [0,1,2,3,4] corresponding to [A,B,C,D,F] from our encoding
    pA, pB, pC, pD, pF = [p * 100 for p in probs]  # convert to percentages
    print(f"{code:8}{pA:5.1f}%{pB:5.1f}%{pC:5.1f}%{pD:5.1f}%{pF:5.1f}%")
