# This script parses a class schedule, identifies the student's major, collects course and instructor info,
# prompts for student GPA and prerequisite grades, trains a Bayesian Network (Naive Bayes) model on synthetic data,
# and outputs the grade probability distribution for each course along with the model's accuracy.

# Section 1: Parse weekly class schedule and identify enrolled courses and the student's major
from collections import Counter
import pandas as pd
import numpy as np
from sklearn.naive_bayes import CategoricalNB
from sklearn.model_selection import train_test_split

# ---------- Section 1: Parse Schedule ----------
try:
    with open("csp/hypothetical_schedule.txt", "r") as f:
        schedule_lines = f.readlines()
except FileNotFoundError:
    print("Error: 'hypothetical_schedule.txt' not found.")
    exit(1)

courses_list = []
course_name_map = {}
instructor_map = {}
day_map = {}
current_day = None

for line in schedule_lines:
    line = line.strip()
    if not line:
        continue
    if line.endswith(":"):
        current_day = line.replace(":", "")
        continue
    if "Room:" in line:
        continue
    if " - " in line:
        time_part, rest = line.split(" - ", 1)
        if ":" in rest:
            code, rest = rest.split(":", 1)
            code = code.strip()
            course_name = rest.split("(")[0].strip()
            instr = rest.split("(")[1].replace(")", "").strip() if "(" in rest else ""
            courses_list.append(code)
            course_name_map[code] = course_name
            instructor_map[code] = instr
            day_map.setdefault(code, []).append(current_day)

course_counts = Counter(courses_list)
enrolled_courses = list(course_counts.keys())

major_course = next((code for code, count in course_counts.items() if count == 3), max(course_counts, key=course_counts.get))
major_prefix = "".join([c for c in major_course if not c.isdigit()])
student_major_prefix = major_prefix

print(f"Identified courses: {enrolled_courses}")
print(f"The course '{major_course}' appears {course_counts[major_course]} times. Assuming major in '{student_major_prefix}' department.")

# ---------- Section 2: Load Data ----------
courses_df = pd.read_csv("csp/course_csp_limited.csv")
prof_df = pd.read_csv("data/prof_qaulity_info.csv")

# Extract all unique department prefixes to be valid majors
all_course_codes = courses_df['course_code'].dropna().unique()
major_categories = sorted(set("".join([c for c in code if not c.isdigit()]) for code in all_course_codes))
major_categories.append('Other')  # fallback category

class_size_map = dict(zip(courses_df['course_code'], courses_df['num_students']))
prof_info_map = {}
for _, row in prof_df.iterrows():
    name = row['name']
    try:
        rating_val = float(row['rating_val'])
    except:
        rating_val = 3.0
    try:
        diff_val = float(row['diff_level'])
    except:
        diff_val = 3.0
    prof_info_map[name] = (rating_val, diff_val)

course_info = {}
for code in enrolled_courses:
    instr_name = instructor_map.get(code, "")
    rating_val, diff_val = prof_info_map.get(instr_name, (3.0, 3.0))
    course_info[code] = {
        "name": course_name_map.get(code, ""),
        "instructor": instr_name,
        "class_size": class_size_map.get(code, "Unknown"),
        "prof_rating": rating_val,
        "prof_difficulty": diff_val,
        "days": day_map.get(code, ["N/A"])
    }
# ---------- Section 3: User Inputs ----------
print("\nLet's learn more about you!")
morning_pref = input("Are you a morning person or a night owl? (Enter 'morning' or 'night'): ").lower()
friend_map = {}
for course in enrolled_courses:
    response = input(f"Do you have friends in {course}? (yes/no): ").lower()
    friend_map[course] = response == "yes"

print("\n\U0001F4D6 Your Schedule for Next Semester:")
print(f"You are a {student_major_prefix} Major\n")
for code in enrolled_courses:
    info = course_info.get(code, {})
    print(f"\U0001F4D8 {code} - {info['name']}")
    print(f"\U0001F468‍\U0001F3EB Instructor: {info['instructor']}")
    print(f"⭐ Rating: {info['prof_rating']} | 💣 Difficulty: {info['prof_difficulty']}")
    print(f"📅 Days: {', '.join(set(info['days']))}")
    print(f"🏫 Class Size: {info['class_size']} students")
    print(f"🧠 Related to Your Major: {'Yes' if code.startswith(student_major_prefix) else 'No'}")
    print("".ljust(40, "-"))

while True:
    try:
        user_gpa = float(input("Enter your current GPA (0.0 - 4.3): "))
        if 0 <= user_gpa <= 4.3:
            break
    except ValueError:
        pass
    print("Please enter a valid GPA between 0.0 and 4.3.")

# ---------- Section 4: Distinct Prereqs ----------
prereq_df = pd.read_csv("bayesian/complete_courses.csv", usecols=['course_code', 'prereq_codes'])
prereq_lookup = {}
for _, row in prereq_df.iterrows():
    code = row['course_code'].replace(" ", "")
    prereqs = [p.strip() for p in str(row['prereq_codes']).split(',') if p.strip()]
    prereq_lookup[code] = prereqs

prereq_avg_cat = {code: 'None' for code in enrolled_courses}
prereq_cache = {}

for code in enrolled_courses:
    number_part = "".join([c for c in code if c.isdigit()]) or "0"
    if int(number_part) >= 200:
        prereqs = prereq_lookup.get(code, [])
        grades = []
        for prereq in prereqs:
            if prereq not in prereq_cache:
                while True:
                    try:
                        grade = int(input(f"Enter your grade (0-100) in prerequisite course {prereq}: "))
                        if 0 <= grade <= 100:
                            prereq_cache[prereq] = grade
                            break
                    except ValueError:
                        pass
                    print("Please enter a number between 0 and 100.")
            grades.append(prereq_cache[prereq])
        if grades:
            avg = sum(grades)/len(grades)
            if avg >= 80:
                prereq_avg_cat[code] = 'High'
            elif avg >= 60:
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
user_X = []
for code in enrolled_courses:
    info = course_info.get(code, {})
    # Professor difficulty category (thresholds: <=2.5 Low, >=3.6 High, otherwise Medium)
    diff_val = info.get("prof_difficulty")
    if diff_val is None or np.isnan(diff_val):
        diff_cat = 'Medium'
    else:
        diff_cat = 'Low' if diff_val <= 2.5 else ('High' if diff_val >= 3.6 else 'Medium')
    # Professor quality (rating) category (thresholds: <3.0 Low, >=4.0 High, otherwise Medium)
    rating_val = info.get("prof_rating")
    if rating_val is None or np.isnan(rating_val):
        qual_cat = 'Medium'
    else:
        qual_cat = 'Low' if rating_val < 3.0 else ('High' if rating_val >= 4.0 else 'Medium')
    # Class size category (<=50 Small, >100 Large, otherwise Medium)
    size_val = info.get("class_size")
    if size_val is None or size_val == "Unknown":
        class_cat = 'Medium'
    else:
        try:
            size_val = int(size_val)
            class_cat = 'Small' if size_val <= 50 else ('Large' if size_val > 100 else 'Medium')
        except (ValueError, TypeError):
            class_cat = 'Medium'
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
    
    # IMPORTANT: Make sure major_cat is in the major_map keys
    if major_cat not in major_map:
        major_cat = 'Other'

    # Major-course match (Yes if course prefix matches major prefix, else No)
    course_prefix = "".join([c for c in code if not c.isdigit()])
    match_cat = 'Yes' if course_prefix == student_major_prefix else 'No'
    # Participation category based on friends
    if friend_map[code]:
        part_cat = 'High'
    else:
        part_cat = 'Medium'
    # Prerequisite average grade category from earlier input
    prereq_cat = prereq_avg_cat.get(code, 'None')
    
    # IMPORTANT: Ensure all categories are valid before encoding
    if diff_cat not in difficulty_map: diff_cat = 'Medium'
    if qual_cat not in quality_map: qual_cat = 'Medium'
    if class_cat not in class_map: class_cat = 'Medium'
    if timing_cat not in timing_map: timing_cat = 'Afternoon'
    if gpa_cat not in gpa_map: gpa_cat = 'Medium'
    if match_cat not in match_map: match_cat = 'No'
    if part_cat not in participation_map: part_cat = 'Medium'
    if prereq_cat not in prereq_map: prereq_cat = 'None'
    
    # Append the encoded feature vector for this course
    user_X.append([
        difficulty_map[diff_cat],
        quality_map[qual_cat],
        class_map[class_cat],
        timing_map[timing_cat],
        gpa_map[gpa_cat],
        major_map[major_cat],  # No fallback needed as we've already sanitized
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
