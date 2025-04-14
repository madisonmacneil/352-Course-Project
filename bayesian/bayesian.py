#  Imports and Data Loading
import re
import pyro
import pyro.distributions as dist
import torch
import numpy as np
import pandas as pd
import json

# Set random seed for reproducibility
pyro.set_rng_seed(0)

# Load course information (course code, name, class size, professor)
courses_df = pd.read_csv('csp/course_csp.csv')
# Load professor quality information (name, rating, difficulty, etc.)
prof_df = pd.read_csv('data/prof_qaulity_info.csv')
prof_df['diff_level'] = pd.to_numeric(prof_df['diff_level'], errors='coerce')  # convert 'N/A' to NaN for diff_level

courses_full_df = pd.read_csv('bayesian/complete_courses.csv', usecols=['course_code', 'prereq_codes'])

# Load mapping from majors to broad categories (category_map)
category_map = {}
with open('bayesian/category_map.txt') as f:
    for line in f:
        if ':' in line:
            # print(line)
            major_name, category = line.strip().split(': ')
            category_map[major_name] = category

# Load major-course success matrix (probability of grades given major & course category)
with open('bayesian/major_course_success_matrix.json') as f:
    major_course_matrix = json.load(f)

# Prepare a lookup for department name from department code using complete_courses data
dept_df = pd.read_csv('bayesian/complete_courses.csv', usecols=['department_name', 'department_code']).drop_duplicates()
dept_to_name = {code: name for code, name in zip(dept_df['department_code'], dept_df['department_name'])}

#  Define helper functions for conditional probability tables (heuristic-based)
def get_prereqs(course_code):
    normalized_code = course_code.replace(" ", "").upper()
    courses_full_df['normalized_code'] = courses_full_df['course_code'].str.replace(" ", "").str.upper()
    row = courses_full_df[courses_full_df['normalized_code'] == normalized_code]
    if row.empty:
        return []
    raw = row.iloc[0]['prereq_codes']
    if pd.isna(raw):
        return []
    return re.findall(r'([A-Z]{3,4}\s?\d{3})', raw)

def ask_user_for_prereq_grades(prereqs):
    print("\nEnter your grades for the following prerequisites:")
    grades = {}
    for c in prereqs:
        g = input(f"Grade for {c}: ").strip()
        grades[c] = letter_to_gpa(g)
    return grades

def get_prof_of_course(code):
    row = courses_df[courses_df['course_code'] == code]
    return row.iloc[0]['prof'] if not row.empty else None

def get_gpa_with_prof(prereq_grades, prof):
    shared = []
    for course, gpa in prereq_grades.items():
        if get_prof_of_course(course) == prof:
            shared.append(gpa)
    return np.mean(shared) if shared else None


def letter_to_gpa(letter):
    """Convert letter or numeric grades to GPA (0-4 scale)."""
    letter = letter.strip().upper()
    mapping = {'A': 4.0, 'B': 3.0, 'C': 2.0, 'D': 1.0, 'F': 0.0}

    if letter in mapping:
        return mapping[letter]
    else:
        # Handle numeric input (0-100 scale)
        try:
            numeric_grade = float(letter)
            if numeric_grade >= 85:
                return 4.0  # A
            elif numeric_grade >= 75:
                return 3.0  # B
            elif numeric_grade >= 65:
                return 2.0  # C
            elif numeric_grade >= 50:
                return 1.0  # D
            else:
                return 0.0  # F
        except ValueError:
            return 0.0  # If it's neither letter nor numeric, default to 0

def grade_percent_to_vec(grade_percent):
    """Maps numerical grade percent (0-100) to aptitude vector."""
    if grade_percent >= 90:
        return np.array([0.05, 0.15, 0.80])  # High Aptitude
    elif grade_percent >= 80:
        return np.array([0.10, 0.60, 0.30])  # Good Aptitude
    elif grade_percent >= 65:
        return np.array([0.30, 0.60, 0.10])  # Medium Aptitude
    elif grade_percent >= 50:
        return np.array([0.60, 0.30, 0.10])  # Low Aptitude
    else:
        return np.array([0.80, 0.15, 0.05])  # Very Low Aptitude

def get_subject_aptitude_probs(inputs):
    # Conditional probability for Subject Aptitude based on inputs.
    gpa_faculty = inputs.get('gpa_in_faculty')
    prereq_grade = inputs.get('prerequisite_grade')
    overall_gpa = inputs.get('overall_gpa')
    major_cat = inputs.get('major_category')
    course_cat = inputs.get('course_category')

    # CPT for subject aptitude
    base_probs = [0.2, 0.4, 0.4]  # Default to a balanced distribution (Low, Medium, High)

    # Example conditions for aptitude change
    if prereq_grade is not None:
        if prereq_grade >= 85:
            base_probs = [0.05, 0.25, 0.70]
        elif prereq_grade >= 75:
            base_probs = [0.10, 0.60, 0.30]
        elif prereq_grade >= 65:
            base_probs = [0.20, 0.50, 0.30]
        else:
            base_probs = [0.50, 0.40, 0.10]

    if gpa_faculty is not None:
        if gpa_faculty >= 3.7:
            base_probs = [0.10, 0.30, 0.60]
        elif gpa_faculty >= 3.0:
            base_probs = [0.20, 0.50, 0.30]
        else:
            base_probs = [0.40, 0.40, 0.20]

    # Normalize probabilities
    base_probs = np.array(base_probs) / np.sum(base_probs)
    return base_probs.tolist()

def get_student_strength_probs(inputs):
    """Heuristic CPT for Student Strength (Low/Medium/High) given overall ability and commitments."""
    overall_gpa = inputs.get('overall_gpa')
    course_load = inputs.get('course_load')  # number of courses taken concurrently
    job_status = inputs.get('job_status')    # 'None', 'Part-time', 'Full-time'

    # Start with overall GPA as base indicator of academic strength
    if overall_gpa is None:
        base_strength = 'Medium'
    elif overall_gpa >= 3.7:
        base_strength = 'High'
    elif overall_gpa >= 3.0:
        base_strength = 'Medium'
    else:
        base_strength = 'Low'
    
    # Adjust for course load and job status
    if course_load:
        if course_load > 5:  # overload of courses
            base_strength = 'Medium' if base_strength == 'High' else 'Low'
        elif course_load <= 4:  # light load
            base_strength = 'High' if base_strength == 'Medium' else base_strength
    
    if job_status == 'part-time':
        base_strength = 'Medium' if base_strength == 'High' else 'Low'
    elif job_status == 'full-time':
        base_strength = 'Low'

    # Return probabilities based on strength
    if base_strength == 'High':
        return [0.10, 0.20, 0.70]
    elif base_strength == 'Medium':
        return [0.10, 0.80, 0.10]
    else:
        return [0.70, 0.20, 0.10]

def get_course_difficulty_probs(inputs):
    """Heuristic CPT for Course Difficulty based on course attributes."""
    course_code = inputs.get('course_code', '')
    prof_diff = inputs.get('prof_diff')

    # Base difficulty based on course level
    level_num = int(course_code[-3:])  # Assuming course code ends with level number

    base_difficulty = 'Medium'
    if level_num >= 300:
        base_difficulty = 'High'
    elif level_num >= 200:
        base_difficulty = 'Medium'
    else:
        base_difficulty = 'Low'

    # Adjust based on professor's difficulty rating
    if prof_diff is not None:
        if prof_diff >= 4.0:
            base_difficulty = 'High'
        elif prof_diff <= 2.5:
            base_difficulty = 'Low'

    if base_difficulty == 'High':
        return [0.0, 0.3, 0.7]  # Likely hard
    elif base_difficulty == 'Medium':
        return [0.1, 0.8, 0.1]
    else:
        return [0.7, 0.3, 0.0]

def get_class_participation_probs(inputs):
    """CPT for Class Participation (Low/Medium/High) based on class size and friends presence.
    If class is small or the student has friends in class, participation tends to be higher."""
    class_size = inputs.get('class_size', 0)
    has_friends = inputs.get('friends_in_class', False)
    # Large classes -> generally low participation
    if class_size > 100:
        if has_friends:
            return [0.60, 0.40, 0.00]  # friend might slightly encourage engagement
        else:
            return [0.90, 0.10, 0.00]
    # Medium classes (50-100 students)
    elif class_size > 50:
        if has_friends:
            return [0.30, 0.60, 0.10]
        else:
            return [0.50, 0.50, 0.00]
    # Small classes (<50 students)
    else:
        if has_friends:
            return [0.00, 0.40, 0.60]  # likely to participate a lot
        else:
            return [0.10, 0.70, 0.20]

def get_attendance_probs(inputs):
    """CPT for Makes it to Class (attendance Low/Medium/High) based on user input or circadian fit.
    If a specific attendance percentage is given, map it to a category with high certainty.
    Otherwise, use EarlyBird/NightOwl vs class timing/day to estimate attendance."""
    att_percent = inputs.get('attendance_percent')

    vec = []

    if att_percent is not None:
        # Map attendance percentage directly to category (with minimal uncertainty)
        # High if >=66%, Medium if 33-65%, Low if <33%
        percent = att_percent if att_percent <= 100 else att_percent * 100  # handle if given 0-1 or 0-100
        if percent >= 80:
            vec = [0.0, 0.1, 0.9]   # almost certainly High attendance
        elif percent >= 40:
             vec = [0.1, 0.8, 0.1]   # mostly Medium
        else:
            vec = [0.9, 0.1, 0.0]   # almost certainly Low attendance
    # If no explicit percentage, use EarlyBird/NightOwl preference and class time/day
    early_bird = inputs.get('early_bird', False)
    class_time = inputs.get('class_time', None)
    # Default base probability
    # Adjust for class timing relative to student's preference
    if class_time:

        for t in class_time:
            time_str = str(t)
            m = re.match(r'(\d+):', time_str)
            hour = int(m.group(1))
            # If class is early morning and user is night owl -> likely low attendance
            if hour is not None:
                if hour < 12 and not early_bird:
                    vec = vec + np.array([0.8, 0.2, 0.0])
                # early bird is likely to attend early class
                elif hour < 12 and early_bird:
                    vec = vec + np.array([0.0, 0.2, 0.8])
                # afternoon class likely to be attend by night owl 
                elif hour >= 1 and not early_bird:
                    vec = vec + np.array([0.0, 0.2, 0.8])

        vec = [x / len(class_time) for x in vec]
    return vec
def get_participation_marks_probs(attendance_level, participation_level):
    """CPT for Participation Marks (Low/Medium/High) based on attendance and class participation levels.
    Assumes full participation marks only if both attendance and participation are high, 
    and no marks if either is very low."""
    # Map attendance and participation to Low/Medium/High index if given as value
    if isinstance(attendance_level, str):
        att_idx = {'Low': 0, 'Medium': 1, 'High': 2}[attendance_level]
    else:
        att_idx = int(attendance_level)
    if isinstance(participation_level, str):
        part_idx = {'Low': 0, 'Medium': 1, 'High': 2}[participation_level]
    else:
        part_idx = int(participation_level)
    # Determine participation marks category
    if att_idx == 2 and part_idx == 2:
        return [0.0, 0.1, 0.9]   # likely High marks (attended and participated fully)
    elif att_idx == 0 or part_idx == 0:
        return [0.9, 0.1, 0.0]   # likely Low marks (missed too many classes or no participation)
    else:
        return [0.1, 0.8, 0.1]   # Medium marks in most other cases

#  Define the Bayesian Network model using Pyro
def bayes_net_model(inputs):
    # Sample Subject Aptitude (latent) given major background and performance
    p_apt = torch.tensor(get_subject_aptitude_probs(inputs), dtype=torch.float)
    subject_apt = pyro.sample("subject_aptitude", dist.Categorical(p_apt))
    # Sample Course Difficulty (latent) given course attributes
    p_diff = torch.tensor(get_course_difficulty_probs(inputs), dtype=torch.float)
    course_difficulty = pyro.sample("course_difficulty", dist.Categorical(p_diff))
    # Sample Student Strength (latent) given student profile
    p_strength = torch.tensor(get_student_strength_probs(inputs), dtype=torch.float)
    student_strength = pyro.sample("student_strength", dist.Categorical(p_strength))
    # Sample Makes it to Class (attendance) node. If user provided an attendance percentage, treat as observed.
    if inputs.get('attendance_percent') is not None:
        # Map percentage to attendance level index (0=Low,1=Med,2=High)
        pct = inputs['attendance_percent']
        if pct <= 1: 
            pct *= 100
        att_level = 2 if pct >= 66 else 1 if pct >= 33 else 0
        pyro.sample("attendance", dist.Delta(torch.tensor(att_level)), obs=torch.tensor(att_level))
        attendance = att_level  # use as value in subsequent logic
    else:
        p_att = torch.tensor(get_attendance_probs(inputs), dtype=torch.float)
        attendance = pyro.sample("attendance", dist.Categorical(p_att))
    # Sample Class Participation (latent) given class size and friends
    p_part = torch.tensor(get_class_participation_probs(inputs), dtype=torch.float)
    class_participation = pyro.sample("class_participation", dist.Categorical(p_part))
    # Sample Participation Marks (latent) given attendance and participation
    p_marks = torch.tensor(get_participation_marks_probs(attendance, class_participation), dtype=torch.float)
    participation_marks = pyro.sample("participation_marks", dist.Categorical(p_marks))
    # Finally, sample Course Grade given all influencing factors (using a heuristic CPT via factor multipliers)
    # Get base grade distribution from major-course success matrix
    major_cat = inputs['major_category']
    course_cat = inputs['course_category']
    base_dist = major_course_matrix.get(f"{major_cat}|{course_cat}")
    if base_dist is None:
        # If no direct data, assume an average distribution
        base_probs = torch.tensor([0.2, 0.2, 0.2, 0.2, 0.2], dtype=torch.float)
    else:
        base_probs = torch.tensor([base_dist['A'], base_dist['B'], base_dist.get('C',0), base_dist.get('D',0), base_dist.get('F',0)], dtype=torch.float)
    # Define multiplicative factors for each influencing node (heuristic adjustments)
    apt_levels = ['Low', 'Medium', 'High']
    diff_levels = ['Low', 'Medium', 'High']
    str_levels = ['Low', 'Medium', 'High']
    marks_levels = ['Low', 'Medium', 'High']
    # Map sampled indices to level names
    apt_level = apt_levels[int(subject_apt.item())]
    diff_level = diff_levels[int(course_difficulty.item())]
    str_level = str_levels[int(student_strength.item())]
    marks_level = marks_levels[int(participation_marks.item())]
    # Professor quality factor based on RateMyProf rating (if available)
    prof_rating = inputs.get('prof_rating')
    if prof_rating is not None:
        if prof_rating >= 4.0:
            prof_factor = torch.tensor([1.1, 1.05, 1.0, 0.95, 0.90], dtype=torch.float)
        elif prof_rating < 3.0:
            prof_factor = torch.tensor([0.90, 0.95, 1.0, 1.05, 1.10], dtype=torch.float)
        else:
            prof_factor = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0], dtype=torch.float)
    else:
        prof_factor = torch.ones(5)  # neutral if no info
    # Factors for aptitude, difficulty, strength, and participation marks (defined as torch tensors)
    apt_factor = {
        'High': torch.tensor([1.10, 1.05, 1.00, 0.90, 0.80]),  # High aptitude -> boosts higher grades
        'Medium': torch.tensor([1.00, 1.00, 1.00, 1.00, 1.00]),
        'Low': torch.tensor([0.80, 0.90, 1.00, 1.10, 1.20])    # Low aptitude -> shifts toward lower grades
    }
    diff_factor = {
        'High': torch.tensor([0.70, 0.85, 1.00, 1.15, 1.30]),  # High difficulty -> more low grades
        'Medium': torch.tensor([1.00, 1.00, 1.00, 1.00, 1.00]),
        'Low': torch.tensor([1.30, 1.15, 1.00, 0.85, 0.70])    # Low difficulty -> easier A/B grades
    }
    strength_factor = {
        'High': torch.tensor([1.10, 1.05, 1.00, 0.90, 0.80]),
        'Medium': torch.tensor([1.00, 1.00, 1.00, 1.00, 1.00]),
        'Low': torch.tensor([0.80, 0.90, 1.00, 1.10, 1.20])
    }
    part_marks_factor = {
        'High': torch.tensor([1.05, 1.05, 1.00, 0.95, 0.95]),  # Full participation marks slightly boost grade
        'Medium': torch.tensor([1.00, 1.00, 1.00, 1.00, 1.00]),
        'Low': torch.tensor([0.95, 0.95, 1.00, 1.05, 1.10])    # Missing participation marks can drop the grade
    }
    # Multiply base probabilities by all factor adjustments
    adjusted_probs = base_probs * apt_factor[apt_level] * diff_factor[diff_level] * strength_factor[str_level] * part_marks_factor[marks_level] * prof_factor
    # Normalize to get a proper probability distribution
    grade_probs = adjusted_probs / adjusted_probs.sum()
    # Sample final course grade (categorical over [A, B, C, D, F])
    course_grade = pyro.sample("course_grade", dist.Categorical(grade_probs))
    return course_grade  # return the sampled grade index (0=A, 1=B, ...)

def explain_inference(course_code, user_profile):
    # Get info like before
    course_row = courses_df[courses_df['course_code'] == course_code]
    if course_row.empty:
        raise ValueError(f"Course code {course_code} not found.")
    class_size = int(course_row.iloc[0]['num_students'])
    professor_name = course_row.iloc[0]['prof']
    prof_row = prof_df[prof_df['name'] == professor_name]
    prof_rating = float(prof_row.iloc[0]['rating_val']) if not prof_row.empty else None
    prof_diff = float(prof_row.iloc[0]['diff_level']) if not prof_row.empty and pd.notna(prof_row.iloc[0]['diff_level']) else None

    prereqs = get_prereqs(course_code)
    print(f"DEBUG: Prereqs for {course_code} = {prereqs}")
    prerequisite_grades = ask_user_for_prereq_grades(prereqs) if prereqs else {}
    avg_prereq_gpa = np.mean(list(prerequisite_grades.values())) if prerequisite_grades else None
    gpa_with_prof = get_gpa_with_prof(prerequisite_grades, professor_name)

    user_major = user_profile.get('major')
    if user_major in category_map:
        major_category = category_map[user_major]
    else:
        major_category = user_major

    # Course category
    import re
    dept_code_match = re.match(r'^[A-Za-z]+', course_code)
    course_category = None
    if dept_code_match:
        dept_code = dept_code_match.group(0)
        dept_name = dept_to_name.get(dept_code)
        course_category = category_map.get(dept_name, 'Interdisciplinary')

    # Model inputs
    inputs = {
        'major_category': major_category,
        'overall_gpa': user_profile.get('overall_gpa'),
        'gpa_in_faculty': user_profile.get('gpa_in_faculty'),
        'gpa_with_prof': gpa_with_prof,
        'early_bird': user_profile.get('early_bird', False),
        'attendance_percent': user_profile.get('attendance_percent'),
        'class_time': user_profile.get('class_time'),
        'friends_in_class': user_profile.get('friends_in_class', False),
        'course_load': user_profile.get('course_load'),
        'job_status': user_profile.get('job_status'),
        'prerequisite_grade': avg_prereq_gpa,
        'class_size': class_size,
        'course_code': course_code,
        'prof_rating': prof_rating,
        'prof_diff': prof_diff
    }

    # Get probabilities
    base_dist = major_course_matrix.get(f"{major_category}|{course_category}")
    base_probs = torch.tensor([base_dist.get(k, 0) for k in ['A', 'B', 'C', 'D', 'F']], dtype=torch.float) \
        if base_dist else torch.tensor([0.2]*5, dtype=torch.float)

    apt = get_subject_aptitude_probs(inputs)
    diff = get_course_difficulty_probs(inputs)
    strength = get_student_strength_probs(inputs)
    attendance = get_attendance_probs(inputs)
    participation = get_class_participation_probs(inputs)
    attendance_idx = torch.tensor([0, 1, 2])[torch.tensor(attendance).argmax()]
    participation_idx = torch.tensor([0, 1, 2])[torch.tensor(participation).argmax()]
    part_marks = get_participation_marks_probs(attendance_idx, participation_idx)

    apt_levels = ['Low', 'Medium', 'High']
    diff_levels = ['Low', 'Medium', 'High']
    str_levels = ['Low', 'Medium', 'High']
    marks_levels = ['Low', 'Medium', 'High']

    apt_idx = torch.tensor(apt).argmax().item()
    diff_idx = torch.tensor(diff).argmax().item()
    str_idx = torch.tensor(strength).argmax().item()
    marks_idx = torch.tensor(part_marks).argmax().item()

    apt_label = apt_levels[apt_idx]
    diff_label = diff_levels[diff_idx]
    str_label = str_levels[str_idx]
    marks_label = marks_levels[marks_idx]

    # Multiplier dictionaries
    apt_factor = {
        'High': torch.tensor([1.10, 1.05, 1.00, 0.90, 0.80]),
        'Medium': torch.tensor([1.00] * 5),
        'Low': torch.tensor([0.80, 0.90, 1.00, 1.10, 1.20])
    }
    diff_factor = {
        'High': torch.tensor([0.70, 0.85, 1.00, 1.15, 1.30]),
        'Medium': torch.tensor([1.00] * 5),
        'Low': torch.tensor([1.30, 1.15, 1.00, 0.85, 0.70])
    }
    strength_factor = {
        'High': torch.tensor([1.10, 1.05, 1.00, 0.90, 0.80]),
        'Medium': torch.tensor([1.00] * 5),
        'Low': torch.tensor([0.80, 0.90, 1.00, 1.10, 1.20])
    }
    part_marks_factor = {
        'High': torch.tensor([1.05, 1.05, 1.00, 0.95, 0.95]),
        'Medium': torch.tensor([1.00] * 5),
        'Low': torch.tensor([0.95, 0.95, 1.00, 1.05, 1.10])
    }
    if prof_rating is not None:
        if prof_rating >= 4.0:
            prof_factor = torch.tensor([1.1, 1.05, 1.0, 0.95, 0.90])
        elif prof_rating < 3.0:
            prof_factor = torch.tensor([0.90, 0.95, 1.0, 1.05, 1.10])
        else:
            prof_factor = torch.tensor([1.0] * 5)
    else:
        prof_factor = torch.tensor([1.0] * 5)

    # Apply adjustments
    adjusted = base_probs.clone()
    adjusted *= apt_factor[apt_label]
    adjusted *= diff_factor[diff_label]
    adjusted *= strength_factor[str_label]
    adjusted *= part_marks_factor[marks_label]
    adjusted *= prof_factor
    final = adjusted / adjusted.sum()

    # Print explanation
    print(f"\n--- Grade Prediction Explanation for {course_code} ---")
    print(f"Base grade distribution (major-course match): {dict(zip(['A','B','C','D','F'], base_probs.numpy().round(3)))}\n")
    print(f"Subject Aptitude: {apt_label} -> Multiplier {apt_factor[apt_label].tolist()}")
    print(f"Course Difficulty: {diff_label} -> Multiplier {diff_factor[diff_label].tolist()}")
    print(f"Student Strength: {str_label} -> Multiplier {strength_factor[str_label].tolist()}")
    print(f"Participation Marks: {marks_label} -> Multiplier {part_marks_factor[marks_label].tolist()}")
    print(f"Prof Rating: {prof_rating} -> Multiplier {prof_factor.tolist()}")
    print(f"\nFinal adjusted distribution:")

    print("\n--- Course and Professor Context ---")
    print(f"Course Title     : {course_row.iloc[0]['course_code']}")
    print(f"Class Size       : {class_size}")
    print(f"Instructor       : {professor_name}")
    print(f"RateMyProf Rating: {prof_rating}")
    print(f"Difficulty Rating: {prof_diff}")
    print(f"Class Time       : {user_profile.get('class_time')}")

    for g, p in zip(['A','B','C','D','F'], final.tolist()):
        print(f"{g}: {p:.2f}")


#  Example usage
example_profile = {
    'major': 'Computer Science',      # Student's major
    'overall_gpa': 4.2,               # Overall GPA
    'gpa_in_faculty': 4.3,            # GPA in the faculty of the course (if applicable)
    'early_bird': False,             # The student is a night owl (not an early bird)
    'attendance_percent': 50,         # Expects to attend ~60% of classes
    'class_time': ['8:30', '12:30', '10:30'],        # Course scheduled time
    'friends_in_class': True,        # No friends in this class
    'course_load': 5,                 # Taking 5 courses this semester
    'job_status': None,        # Has a part-time job
}
course_code = "CISC423"  # Selected course code (e.g., General Chemistry)
explain_inference(course_code, example_profile)
