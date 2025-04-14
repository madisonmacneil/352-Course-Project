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
    
def letter_to_percent(letter):
    """Convert a letter grade to a percentage (middle of the range)"""
    letter = letter.strip().upper()
    mapping = {
        'A+': 95,  # 90-100
        'A': 87,   # 85-89.9
        'A-': 82,  # 80-84.9
        'B+': 78,  # 77-79.9
        'B': 75,   # 73-76.9
        'B-': 71,  # 70-72.9
        'C+': 68,  # 67-69.9
        'C': 65,   # 63-66.9
        'C-': 61,  # 60-62.9
        'D+': 58,  # 57-59.9
        'D': 55,   # 53-56.9
        'D-': 51,  # 50-52.9
        'F': 40    # 0-49.9 (arbitrary middle value)
    }
    
    if letter in mapping:
        return mapping[letter]
    else:
        # Handle numeric input (already as 0-100)
        try:
            numeric_grade = float(letter)
            if 0 <= numeric_grade <= 100:
                return numeric_grade
            else:
                return 50  # Default for invalid input
        except ValueError:
            return 50  # Default for invalid input

def percent_to_letter_and_gpa(percent):
    """Convert a percentage grade to letter grade and GPA according to the provided scale"""
    if percent >= 90:
        return 'A+', 4.3
    elif percent >= 85:
        return 'A', 4.0
    elif percent >= 80:
        return 'A-', 3.7
    elif percent >= 77:
        return 'B+', 3.3
    elif percent >= 73:
        return 'B', 3.0
    elif percent >= 70:
        return 'B-', 2.7
    elif percent >= 67:
        return 'C+', 2.3
    elif percent >= 63:
        return 'C', 2.0
    elif percent >= 60:
        return 'C-', 1.7
    elif percent >= 57:
        return 'D+', 1.3
    elif percent >= 53:
        return 'D', 1.0
    elif percent >= 50:
        return 'D-', 0.7
    else:
        return 'F', 0.0

# Define proper CPT for Course Grade given parents
def get_grade_cpt(apt_level, diff_level, strength_level, participation_level):
    """
    A proper CPT for course grade given its parent nodes.
    Returns probability distribution over [A, B, C, D, F]
    """
    # Define indices for cleaner access
    apt_idx = {'Low': 0, 'Medium': 1, 'High': 2}[apt_level]
    diff_idx = {'Low': 0, 'Medium': 1, 'High': 2}[diff_level]
    str_idx = {'Low': 0, 'Medium': 1, 'High': 2}[strength_level]
    part_idx = {'Low': 0, 'Medium': 1, 'High': 2}[participation_level]
    
    # Create a 4D tensor to represent the full CPT
    # CPT dimensions: [aptitude, difficulty, strength, participation_marks, grade]
    
    # Initialize a base CPT with uniform probabilities
    cpt = torch.ones((3, 3, 3, 3, 5))
    
    # High aptitude tends to lead to better grades regardless of other factors
    cpt[2, :, :, :, 0] *= 1.5  # Boost A grades for high aptitude
    cpt[2, :, :, :, 4] *= 0.5  # Reduce F grades for high aptitude
    
    # Low aptitude tends to lead to worse grades
    cpt[0, :, :, :, 0] *= 0.5  # Reduce A grades for low aptitude
    cpt[0, :, :, :, 4] *= 1.5  # Boost F grades for low aptitude
    
    # High difficulty reduces probability of good grades
    cpt[:, 2, :, :, 0] *= 0.6  # Reduce A grades for high difficulty
    cpt[:, 2, :, :, 4] *= 1.4  # Boost F grades for high difficulty
    
    # Low difficulty increases probability of good grades
    cpt[:, 0, :, :, 0] *= 1.4  # Boost A grades for low difficulty
    cpt[:, 0, :, :, 4] *= 0.6  # Reduce F grades for low difficulty
    
    # Student strength affects all grades
    cpt[:, :, 2, :, 0] *= 1.3  # Boost A grades for high strength
    cpt[:, :, 0, :, 4] *= 1.3  # Boost F grades for low strength
    
    # Participation has a smaller but still significant effect
    cpt[:, :, :, 2, 0] *= 1.1  # Slight boost to A grades for high participation
    cpt[:, :, :, 0, 0] *= 0.9  # Slight reduction to A grades for low participation
    
    # Specific combinations with very high/low probabilities
    
    # Ideal student scenario: high aptitude, low difficulty, high strength, high participation
    cpt[2, 0, 2, 2, :] = torch.tensor([0.7, 0.2, 0.07, 0.02, 0.01])
    
    # Worst case scenario: low aptitude, high difficulty, low strength, low participation
    cpt[0, 2, 0, 0, :] = torch.tensor([0.05, 0.15, 0.25, 0.3, 0.25])
    
    # Normalize each conditional distribution to sum to 1
    for a in range(3):
        for d in range(3):
            for s in range(3):
                for p in range(3):
                    cpt[a, d, s, p, :] = cpt[a, d, s, p, :] / cpt[a, d, s, p, :].sum()
    
    # Return the specific conditional distribution for the given parent values
    return cpt[apt_idx, diff_idx, str_idx, part_idx, :]

# Define proper CPT for Participation Marks given Attendance and Class Participation
def get_participation_marks_cpt():
    """
    Returns a 3D tensor representing P(Participation Marks | Attendance, Class Participation)
    Dimensions: [attendance, participation, marks] where each is [Low, Medium, High]
    """
    # Initialize a 3D tensor 
    cpt = torch.zeros((3, 3, 3))
    
    # Define the conditional probabilities
    # High attendance + high participation -> high marks
    cpt[2, 2, 2] = 0.9  # P(Marks=High | Attendance=High, Participation=High)
    cpt[2, 2, 1] = 0.1
    cpt[2, 2, 0] = 0.0
    
    # High attendance + medium participation -> mostly medium marks
    cpt[2, 1, 2] = 0.2
    cpt[2, 1, 1] = 0.7
    cpt[2, 1, 0] = 0.1
    
    # High attendance + low participation -> medium to low marks
    cpt[2, 0, 2] = 0.0
    cpt[2, 0, 1] = 0.4
    cpt[2, 0, 0] = 0.6
    
    # Medium attendance + high participation -> mostly medium marks
    cpt[1, 2, 2] = 0.3
    cpt[1, 2, 1] = 0.6
    cpt[1, 2, 0] = 0.1
    
    # Medium attendance + medium participation -> medium marks
    cpt[1, 1, 2] = 0.1
    cpt[1, 1, 1] = 0.8
    cpt[1, 1, 0] = 0.1
    
    # Medium attendance + low participation -> low marks
    cpt[1, 0, 2] = 0.0
    cpt[1, 0, 1] = 0.3
    cpt[1, 0, 0] = 0.7
    
    # Low attendance + any participation -> mostly low marks
    cpt[0, 2, 2] = 0.0
    cpt[0, 2, 1] = 0.3
    cpt[0, 2, 0] = 0.7
    
    cpt[0, 1, 2] = 0.0
    cpt[0, 1, 1] = 0.2
    cpt[0, 1, 0] = 0.8
    
    cpt[0, 0, 2] = 0.0
    cpt[0, 0, 1] = 0.1
    cpt[0, 0, 0] = 0.9
    
    return cpt

# Define CPT for Class Participation given Class Size and Friends in Class
def get_class_participation_cpt():
    """
    Returns a 3D tensor representing P(Class Participation | Class Size, Friends in Class)
    Dimensions: [class_size_category, has_friends, participation_level]
    """
    # Class size categories: 0=Small (<50), 1=Medium (50-100), 2=Large (>100)
    # Has friends: 0=No, 1=Yes
    # Participation level: 0=Low, 1=Medium, 2=High
    
    cpt = torch.zeros((3, 2, 3))
    
    # Small class, no friends
    cpt[0, 0, 0] = 0.1  # Low participation
    cpt[0, 0, 1] = 0.7  # Medium participation
    cpt[0, 0, 2] = 0.2  # High participation
    
    # Small class, has friends
    cpt[0, 1, 0] = 0.05  # Low participation
    cpt[0, 1, 1] = 0.35  # Medium participation
    cpt[0, 1, 2] = 0.6   # High participation
    
    # Medium class, no friends
    cpt[1, 0, 0] = 0.5  # Low participation
    cpt[1, 0, 1] = 0.5  # Medium participation
    cpt[1, 0, 2] = 0.0  # High participation
    
    # Medium class, has friends
    cpt[1, 1, 0] = 0.3  # Low participation
    cpt[1, 1, 1] = 0.6  # Medium participation
    cpt[1, 1, 2] = 0.1  # High participation
    
    # Large class, no friends
    cpt[2, 0, 0] = 0.9  # Low participation
    cpt[2, 0, 1] = 0.1  # Medium participation
    cpt[2, 0, 2] = 0.0  # High participation
    
    # Large class, has friends
    cpt[2, 1, 0] = 0.6  # Low participation
    cpt[2, 1, 1] = 0.4  # Medium participation
    cpt[2, 1, 2] = 0.0  # High participation
    
    return cpt

# Proper Bayesian Network model using CPTs
def bayes_net_model(inputs):
    # Get class size as category (0=Small, 1=Medium, 2=Large)
    class_size = inputs.get('class_size', 50)
    class_size_cat = 0 if class_size < 50 else 1 if class_size < 100 else 2
    
    # Sample Subject Aptitude (latent) based on inputs
    p_apt = torch.tensor(get_subject_aptitude_probs(inputs), dtype=torch.float)
    subject_apt = pyro.sample("subject_aptitude", dist.Categorical(p_apt))
    
    # Sample Course Difficulty (latent) based on inputs
    p_diff = torch.tensor(get_course_difficulty_probs(inputs), dtype=torch.float)
    course_difficulty = pyro.sample("course_difficulty", dist.Categorical(p_diff))
    
    # Sample Student Strength (latent) based on inputs
    p_strength = torch.tensor(get_student_strength_probs(inputs), dtype=torch.float)
    student_strength = pyro.sample("student_strength", dist.Categorical(p_strength))
    
    # Sample attendance based on inputs
    if inputs.get('attendance_percent') is not None:
        pct = inputs['attendance_percent']
        if pct <= 1: 
            pct *= 100
        att_level = 2 if pct >= 66 else 1 if pct >= 33 else 0
        attendance = pyro.sample("attendance", dist.Delta(torch.tensor(att_level)), obs=torch.tensor(att_level))
    else:
        p_att = torch.tensor(get_attendance_probs(inputs), dtype=torch.float)
        attendance = pyro.sample("attendance", dist.Categorical(p_att))
    
    # Sample Class Participation (latent) given class size and friends
    has_friends = 1 if inputs.get('friends_in_class', False) else 0
    participation_cpt = get_class_participation_cpt()
    p_part = participation_cpt[class_size_cat, has_friends, :]
    class_participation = pyro.sample("class_participation", dist.Categorical(p_part))
    
    # Sample Participation Marks (latent) given attendance and participation
    part_marks_cpt = get_participation_marks_cpt()
    p_marks = part_marks_cpt[int(attendance), int(class_participation), :]
    participation_marks = pyro.sample("participation_marks", dist.Categorical(p_marks))
    
    # Map indices to level names for use in grade CPT
    apt_levels = ['Low', 'Medium', 'High']
    diff_levels = ['Low', 'Medium', 'High']
    str_levels = ['Low', 'Medium', 'High']
    marks_levels = ['Low', 'Medium', 'High']
    
    apt_level = apt_levels[int(subject_apt)]
    diff_level = diff_levels[int(course_difficulty)]
    str_level = str_levels[int(student_strength)]
    marks_level = marks_levels[int(participation_marks)]
    
    # Get grade distribution from CPT
    grade_probs = get_grade_cpt(apt_level, diff_level, str_level, marks_level)
    
    # Apply professor effect as a final modifier
    prof_rating = inputs.get('prof_rating')
    if prof_rating is not None:
        prof_factor = torch.ones(5)
        if prof_rating >= 4.0:
            prof_factor = torch.tensor([1.1, 1.05, 1.0, 0.95, 0.90])
        elif prof_rating < 3.0:
            prof_factor = torch.tensor([0.90, 0.95, 1.0, 1.05, 1.10])
        
        # Apply and normalize
        grade_probs = grade_probs * prof_factor
        grade_probs = grade_probs / grade_probs.sum()
    
    # Sample final course grade
    course_grade = pyro.sample("course_grade", dist.Categorical(grade_probs))
    return course_grade

# Keep or reimplement these existing functions
def get_subject_aptitude_probs(inputs):
    """
    Get conditional probability for Subject Aptitude based on inputs.
    All grade inputs are expected to be on 0-100 scale.
    """
    faculty_grade = inputs.get('faculty_grade')
    prereq_grade = inputs.get('prerequisite_grade')
    overall_grade = inputs.get('overall_grade')
    major_cat = inputs.get('major_category')
    course_cat = inputs.get('course_category')
    
    # Start with a neutral distribution [Low, Medium, High]
    base_vec = np.array([1.0, 1.0, 1.0])
    factors_applied = 0
    
    # Factor 1: Prerequisite grades (strongest effect)
    if prereq_grade is not None:
        factors_applied += 1
        if prereq_grade >= 85:
            prereq_vec = np.array([0.05, 0.25, 0.70])
        elif prereq_grade >= 75:
            prereq_vec = np.array([0.15, 0.55, 0.30])
        elif prereq_grade >= 65:
            prereq_vec = np.array([0.30, 0.50, 0.20])
        else:
            prereq_vec = np.array([0.60, 0.30, 0.10])
        
        # Multiply by this factor
        base_vec *= prereq_vec
    
    # Factor 2: Faculty grades (medium effect)
    if faculty_grade is not None:
        factors_applied += 1
        if faculty_grade >= 85:  # A or higher
            faculty_vec = np.array([0.10, 0.30, 0.60])
        elif faculty_grade >= 75:  # B or higher
            faculty_vec = np.array([0.20, 0.50, 0.30])
        else:  # C or lower
            faculty_vec = np.array([0.40, 0.40, 0.20])
        
        # Multiply by this factor
        base_vec *= faculty_vec
    
    # Factor 3: Major-course match (weaker effect)
    if major_cat is not None and course_cat is not None:
        factors_applied += 1
        if major_cat == course_cat:
            # Good match boosts aptitude
            major_vec = np.array([0.15, 0.35, 0.50])
        else:
            # Poor match lowers aptitude
            major_vec = np.array([0.40, 0.40, 0.20])
        
        # Multiply by this factor
        base_vec *= major_vec
    
    # Factor 4: Overall grade (weakest effect)
    if overall_grade is not None:
        factors_applied += 1
        if overall_grade >= 85:  # A or higher
            overall_vec = np.array([0.15, 0.35, 0.50])
        elif overall_grade >= 75:  # B or higher
            overall_vec = np.array([0.25, 0.50, 0.25])
        else:  # C or lower
            overall_vec = np.array([0.50, 0.35, 0.15])
        
        # Multiply by this factor
        base_vec *= overall_vec
    
    # If no factors were applied, use a reasonable default
    if factors_applied == 0:
        return [0.20, 0.50, 0.30]
    
    # Normalize the final vector to get a proper probability distribution
    base_vec = base_vec / base_vec.sum()
    
    return base_vec.tolist()

def get_course_difficulty_probs(inputs):
    # Improved from your code
    course_code = inputs.get('course_code', '')
    prof_diff = inputs.get('prof_diff')
    
    # Extract course level from code
    import re
    level_num = None
    m = re.search(r'\d+', course_code)
    if m:
        level_num = int(m.group())
    
    # Base difficulty based on course level
    if level_num is not None:
        if level_num >= 300:
            base_probs = [0.1, 0.3, 0.6]  # High difficulty more likely
        elif level_num >= 200:
            base_probs = [0.2, 0.6, 0.2]  # Medium difficulty most likely
        else:
            base_probs = [0.6, 0.3, 0.1]  # Low difficulty more likely
    else:
        base_probs = [0.33, 0.34, 0.33]  # Equal probability if level unknown
    
    # Adjust based on professor difficulty
    if prof_diff is not None:
        if prof_diff >= 4.0:
            # Shift towards higher difficulty
            base_probs = [max(0.0, base_probs[0] - 0.2), 
                          base_probs[1],
                          min(1.0, base_probs[2] + 0.2)]
        elif prof_diff <= 2.5:
            # Shift towards lower difficulty
            base_probs = [min(1.0, base_probs[0] + 0.2),
                          base_probs[1],
                          max(0.0, base_probs[2] - 0.2)]
    
    # Normalize
    base_probs = np.array(base_probs) / np.sum(base_probs)
    return base_probs.tolist()

def get_student_strength_probs(inputs):
    # Improved from your code
    overall_gpa = inputs.get('overall_gpa')
    course_load = inputs.get('course_load')
    job_status = inputs.get('job_status')
    
    # Base probabilities for student strength [Low, Medium, High]
    if overall_gpa is None:
        base_probs = [0.33, 0.34, 0.33]  # Equal probability if GPA unknown
    elif overall_gpa >= 3.7:
        base_probs = [0.1, 0.3, 0.6]  # High strength more likely for high GPA
    elif overall_gpa >= 3.0:
        base_probs = [0.2, 0.6, 0.2]  # Medium strength most likely for average GPA
    else:
        base_probs = [0.6, 0.3, 0.1]  # Low strength more likely for low GPA
    
    # Adjust for course load
    if course_load:
        if course_load > 5:  # Heavy course load
            # Shift probability mass downward (toward lower strength)
            shift = 0.1
            base_probs = [min(1.0, base_probs[0] + shift),
                          base_probs[1],
                          max(0.0, base_probs[2] - shift)]
        elif course_load <= 3:  # Light course load
            # Shift probability mass upward (toward higher strength)
            shift = 0.1
            base_probs = [max(0.0, base_probs[0] - shift),
                          base_probs[1],
                          min(1.0, base_probs[2] + shift)]
    
    # Adjust for job status
    if job_status == 'part-time':
        # Moderate shift downward
        shift = 0.15
        base_probs = [min(1.0, base_probs[0] + shift),
                      base_probs[1],
                      max(0.0, base_probs[2] - shift)]
    elif job_status == 'full-time':
        # Strong shift downward
        shift = 0.25
        base_probs = [min(1.0, base_probs[0] + shift),
                      base_probs[1],
                      max(0.0, base_probs[2] - shift)]
    
    # Normalize
    base_probs = np.array(base_probs) / np.sum(base_probs)
    return base_probs.tolist()

def get_attendance_probs(inputs):
    # Improved from your code
    att_percent = inputs.get('attendance_percent')
    
    if att_percent is not None:
        # Convert to 0-100 scale if needed
        percent = att_percent if att_percent <= 100 else att_percent * 100
        
        # Map to distribution over [Low, Medium, High]
        if percent >= 80:
            return [0.05, 0.15, 0.8]
        elif percent >= 40:
            return [0.15, 0.7, 0.15]
        else:
            return [0.8, 0.15, 0.05]
    
    # If no explicit percentage, use preferences and class time
    early_bird = inputs.get('early_bird', False)
    class_time = inputs.get('class_time', [])
    
    # Default probabilities [Low, Medium, High]
    base_probs = [0.2, 0.6, 0.2]
    
    # Adjust based on class times and early bird preference
    if class_time:
        early_classes = 0
        total_classes = len(class_time)
        
        for t in class_time:
            time_str = str(t)
            import re
            m = re.match(r'(\d+):', time_str)
            if m:
                hour = int(m.group(1))
                if hour < 12:
                    early_classes += 1
        
        # Calculate proportion of early classes
        early_ratio = early_classes / total_classes if total_classes > 0 else 0
        
        # Adjust attendance probabilities based on early bird preference and class timing
        if early_bird and early_ratio > 0.5:  # Early bird with mostly morning classes
            base_probs = [0.1, 0.3, 0.6]  # Higher attendance likely
        elif not early_bird and early_ratio > 0.5:  # Night owl with mostly morning classes
            base_probs = [0.6, 0.3, 0.1]  # Lower attendance likely
        elif early_bird and early_ratio < 0.2:  # Early bird with mostly afternoon classes
            base_probs = [0.3, 0.4, 0.3]  # Moderate attendance
        elif not early_bird and early_ratio < 0.2:  # Night owl with mostly afternoon classes
            base_probs = [0.2, 0.3, 0.5]  # Better attendance
    
    return base_probs


def run_inference(course_code, user_profile, num_samples=1000):
    """
    Run inference on the Bayesian network to predict grade distribution.
    Uses consistent percentage grades (0-100) throughout.
    """
    # Get course information
    course_row = courses_df[courses_df['course_code'] == course_code]
    if course_row.empty:
        raise ValueError(f"Course code {course_code} not found.")
    
    class_size = int(course_row.iloc[0]['num_students'])
    professor_name = course_row.iloc[0]['prof']
    prof_row = prof_df[prof_df['name'] == professor_name]
    prof_rating = float(prof_row.iloc[0]['rating_val']) if not prof_row.empty else None
    prof_diff = float(prof_row.iloc[0]['diff_level']) if not prof_row.empty and pd.notna(prof_row.iloc[0]['diff_level']) else None
    
    # Get prerequisites
    prereqs = get_prereqs(course_code)
    
    # Handle prerequisite grades - either from user input or from parameter
    if hasattr(user_profile, 'direct_prereq_grade'):
        # Direct grade provided for testing/analysis
        avg_prereq_grade = user_profile.direct_prereq_grade
    else:
        # Ask user for grades
        print(f"DEBUG: Prereqs for {course_code} = {prereqs}")
        prereq_grades = {}
        
        if prereqs:
            print("\nEnter your grades for the following prerequisites:")
            for c in prereqs:
                g = input(f"Grade for {c} (0-100 or letter): ").strip()
                prereq_grades[c] = letter_to_percent(g)
            
            # Calculate average
            avg_prereq_grade = sum(prereq_grades.values()) / len(prereq_grades)
        else:
            avg_prereq_grade = None
    
    # Get previous grades with same professor (if any)
    prof_grades = []
    for course, grade in prereq_grades.items() if prereqs else []:
        if get_prof_of_course(course) == professor_name:
            prof_grades.append(grade)
    
    avg_prof_grade = sum(prof_grades) / len(prof_grades) if prof_grades else None
    
    # Get user major and map to category
    user_major = user_profile.get('major')
    major_category = category_map.get(user_major, user_major)
    
    # Get course category
    import re
    dept_code_match = re.match(r'^[A-Za-z]+', course_code)
    course_category = 'Interdisciplinary'  # Default
    if dept_code_match:
        dept_code = dept_code_match.group(0)
        dept_name = dept_to_name.get(dept_code)
        if dept_name in category_map:
            course_category = category_map[dept_name]
    
    # Convert any GPA values in user profile to percentages
    overall_grade = user_profile.get('overall_grade')
    faculty_grade = user_profile.get('faculty_grade')
    
    # Create model inputs using percentage grades throughout
    inputs = {
        'major_category': major_category,
        'course_category': course_category,
        'overall_grade': overall_grade,  # Already in 0-100 scale
        'faculty_grade': faculty_grade,  # Already in 0-100 scale
        'prof_grade': avg_prof_grade,    # Already calculated in 0-100 scale
        'early_bird': user_profile.get('early_bird', False),
        'attendance_percent': user_profile.get('attendance_percent'),
        'class_time': user_profile.get('class_time'),
        'friends_in_class': user_profile.get('friends_in_class', False),
        'course_load': user_profile.get('course_load'),
        'job_status': user_profile.get('job_status'),
        'prerequisite_grade': avg_prereq_grade,  # Already in 0-100 scale
        'class_size': class_size,
        'course_code': course_code,
        'prof_rating': prof_rating,
        'prof_diff': prof_diff
    }
    
    # Print diagnostic information
    print("\nInput profile for grade prediction:")
    print(f"  Prerequisite avg grade: {avg_prereq_grade:.1f}%" if avg_prereq_grade is not None else "  No prerequisites")
    print(f"  Major category: {major_category}, Course category: {course_category}")
    print(f"  Professor rating: {prof_rating}, Difficulty: {prof_diff}")
    
    # Run prediction
    predictive = pyro.infer.Predictive(bayes_net_model, num_samples=num_samples)
    samples = predictive(inputs)
    
    # Calculate grade distribution (0-4 indices representing A, B, C, D, F)
    grade_samples = samples["course_grade"].numpy()
    grade_dist = np.bincount(grade_samples, minlength=5) / num_samples
    
    # Convert indices to letter grades and GPA equivalents
    grade_letters = ['A', 'B', 'C', 'D', 'F']
    # Approximate percentage equivalents for the middle of each grade range
    grade_percentages = [87, 75, 65, 55, 40]
    
    # Create the distribution dictionary
    result = {
        grade_letters[i]: float(grade_dist[i]) for i in range(5)
    }
    
    # Calculate weighted average percentage grade
    weighted_percent = sum(grade_dist[i] * grade_percentages[i] for i in range(5))
    
    # Convert to letter grade and GPA
    final_letter, final_gpa = percent_to_letter_and_gpa(weighted_percent)
    
    # Add the average to the result
    result['avg_percent'] = weighted_percent
    result['avg_letter'] = final_letter
    result['avg_gpa'] = final_gpa
    
    return result

# Example of how to use the model
def example_usage():
    """
    Test the model with default values and show the predicted grade distribution.
    Compatible with the new consistent percentage-based grading system.
    """
    # Base student profile using percentage grades
    base_profile = {
        'major': 'Computer Science',
        'overall_grade': 82,       # B+ (approximately 3.5 GPA)
        'faculty_grade': 85,       # A- (approximately 3.7 GPA)
        'early_bird': False,
        'attendance_percent': 80,
        'class_time': ['10:30', '14:30', '8:30'],
        'friends_in_class': True,
        'course_load': 5,
        'job_status': 'part-time'
    }
    
    course_code = "CISC352"  # You can change this to any course with prerequisites
    
    # Run baseline prediction with default values
    print("\n=============================================")
    print("BASELINE PREDICTION WITH DEFAULT VALUES")
    print("=============================================")
    
    # Use the new consistent inference function
    baseline_prediction = run_inference(course_code, base_profile)
    
    # Display the grade distribution
    print(f"\nBaseline grade distribution for {course_code}:")
    for grade in ['A', 'B', 'C', 'D', 'F']:
        print(f"{grade}: {baseline_prediction[grade]:.3f}")
    
    # Display the average results
    print(f"\nOverall predicted grade:")
    print(f"Average: {baseline_prediction['avg_percent']:.1f}% ({baseline_prediction['avg_letter']})")
    print(f"GPA equivalent: {baseline_prediction['avg_gpa']:.2f}")
    
    # Display interpretation
    letter_grade = baseline_prediction['avg_letter']
    if letter_grade in ['A+', 'A', 'A-']:
        interpretation = "Excellent performance expected"
    elif letter_grade in ['B+', 'B', 'B-']:
        interpretation = "Good performance expected"
    elif letter_grade in ['C+', 'C', 'C-']:
        interpretation = "Average performance expected"
    elif letter_grade in ['D+', 'D', 'D-']:
        interpretation = "Below average performance expected"
    else:
        interpretation = "Poor performance expected, may need significant preparation"
    
    print(f"\nInterpretation: {interpretation}")
    
    # You can also display the factor probabilities to show what influenced this prediction
    print("\nKey factors in this prediction:")
    
    # Get subject aptitude probabilities as an example
    apt_probs = get_subject_aptitude_probs({
        'major_category': base_profile.get('major'), 
        'course_category': 'Computer Science',
        'overall_grade': base_profile.get('overall_grade'),
        'faculty_grade': base_profile.get('faculty_grade'),
        'prerequisite_grade': 75  # Assuming average prerequisite grade
    })
    
    print(f"Subject aptitude: {apt_probs[0]:.2f} Low, {apt_probs[1]:.2f} Medium, {apt_probs[2]:.2f} High")
    print("(Run the sensitivity analysis function for a more detailed breakdown)")
    
def improved_sensitivity_analysis(course_code="CISC352", num_samples=2000):
    """
    Improved sensitivity analysis that uses consistent percentage grades throughout.
    Fixed to use dictionary key instead of attribute.
    """
    print(f"\n=== RUNNING IMPROVED SENSITIVITY ANALYSIS FOR {course_code} ===")
    
    # Base profile for the student - using percentages instead of GPA
    base_profile = {
        'major': 'Computer Science',
        'overall_grade': 50,       # B+ (was 3.5 GPA)
        'faculty_grade': 85,       # A- (was 3.7 GPA)
        'early_bird': False,
        'attendance_percent': 80,
        'class_time': ['10:30', '14:30'],
        'friends_in_class': True,
        'course_load': 5,
        'job_status': 'part-time'
    }
    
    # Store the sensitivity results
    sensitivity_results = []
    
    # Function to run inference with direct prerequisite grade (no prompting)
    def run_with_prereq(prereq_grade):
        # Create a modified version of the run_inference_consistent function that
        # doesn't prompt for grades but uses the provided prereq_grade
        
        # Get course information
        course_row = courses_df[courses_df['course_code'] == course_code]
        if course_row.empty:
            raise ValueError(f"Course code {course_code} not found.")
        
        class_size = int(course_row.iloc[0]['num_students'])
        professor_name = course_row.iloc[0]['prof']
        prof_row = prof_df[prof_df['name'] == professor_name]
        prof_rating = float(prof_row.iloc[0]['rating_val']) if not prof_row.empty else None
        prof_diff = float(prof_row.iloc[0]['diff_level']) if not prof_row.empty and pd.notna(prof_row.iloc[0]['diff_level']) else None
        
        # Get user major and map to category
        user_major = base_profile.get('major')
        major_category = category_map.get(user_major, user_major)
        
        # Get course category
        import re
        dept_code_match = re.match(r'^[A-Za-z]+', course_code)
        course_category = 'Interdisciplinary'  # Default
        if dept_code_match:
            dept_code = dept_code_match.group(0)
            dept_name = dept_to_name.get(dept_code)
            if dept_name in category_map:
                course_category = category_map[dept_name]
        
        # Create model inputs using the direct prerequisite grade
        inputs = {
            'major_category': major_category,
            'course_category': course_category,
            'overall_grade': base_profile.get('overall_grade'),
            'faculty_grade': base_profile.get('faculty_grade'),
            'prof_grade': None,  # Simplified for testing
            'early_bird': base_profile.get('early_bird', False),
            'attendance_percent': base_profile.get('attendance_percent'),
            'class_time': base_profile.get('class_time'),
            'friends_in_class': base_profile.get('friends_in_class', False),
            'course_load': base_profile.get('course_load'),
            'job_status': base_profile.get('job_status'),
            'prerequisite_grade': prereq_grade,  # Use the provided grade
            'class_size': class_size,
            'course_code': course_code,
            'prof_rating': prof_rating,
            'prof_diff': prof_diff
        }
        
        # Print diagnostic information
        print(f"\nTesting with: {course_code}, prereq grade: {prereq_grade}%, "
              f"overall grade: {inputs['overall_grade']}%, faculty grade: {inputs['faculty_grade']}%")
        
        # Check subject aptitude probabilities to debug
        aptitude_probs = get_subject_aptitude_probs(inputs)
        print(f"  Subject aptitude probs [L,M,H]: {np.round(aptitude_probs, 2)}")
        
        # Run prediction with more samples for better stability
        predictive = pyro.infer.Predictive(bayes_net_model, num_samples=num_samples)
        samples = predictive(inputs)
        
        # Calculate grade distribution
        grade_samples = samples["course_grade"].numpy()
        grade_dist = np.bincount(grade_samples, minlength=5) / num_samples
        
        # Grade letters and approximate percentages for each category
        grade_letters = ['A', 'B', 'C', 'D', 'F']
        grade_percentages = [87, 75, 65, 55, 40]
        
        # Create the distribution dictionary
        result = {
            grade_letters[i]: float(grade_dist[i]) for i in range(5)
        }
        
        # Calculate weighted average percentage grade
        weighted_percent = sum(grade_dist[i] * grade_percentages[i] for i in range(5))
        
        # Convert to letter grade and GPA
        final_letter, final_gpa = percent_to_letter_and_gpa(weighted_percent)
        
        # Add the average to the result
        result['avg_percent'] = weighted_percent
        result['avg_letter'] = final_letter
        result['avg_gpa'] = final_gpa
        
        # Print the result
        print(f"  Result: Avg GPA {final_gpa:.2f}, Grade dist: "
              f"A:{result['A']:.2f}, B:{result['B']:.2f}, C:{result['C']:.2f}, "
              f"D:{result['D']:.2f}, F:{result['F']:.2f}")
        
        return result
    
    # ------------------------------------------------------------------------
    # First, establish baseline with default values
    # ------------------------------------------------------------------------
    print("\n=== BASELINE PREDICTION ===")
    baseline_prereq = 75  # B grade
    baseline_result = run_with_prereq(baseline_prereq)
    baseline_avg = baseline_result['avg_gpa']
    print(f"Baseline GPA (with prereq {baseline_prereq}%): {baseline_avg:.2f}")
    
    # ------------------------------------------------------------------------
    # 1. Prerequisite Grade Sensitivity
    # ------------------------------------------------------------------------
    print("\n=== PREREQUISITE GRADE SENSITIVITY ===")
    prereq_grades = [50, 65, 75, 85, 95]  # D-, C, B, A-, A+
    
    for grade in prereq_grades:
        print(f"\nTesting with prerequisite grade: {grade}%")
        result = run_with_prereq(grade)
        
        # Record result
        sensitivity_results.append({
            'factor': 'Prerequisite Grade',
            'value': grade, 
            'avg_gpa': result['avg_gpa'],
            'avg_letter': result['avg_letter'],
            'avg_percent': result['avg_percent'],
            'diff': result['avg_gpa'] - baseline_avg,
            'distribution': {k: v for k, v in result.items() if k in ['A', 'B', 'C', 'D', 'F']}
        })
    
    # ------------------------------------------------------------------------
    # 2. Overall Grade Sensitivity
    # ------------------------------------------------------------------------
    print("\n=== OVERALL GRADE SENSITIVITY ===")
    overall_grades = [65, 75, 82, 90]  # C, B, B+, A+
    
    for grade in overall_grades:
        modified_profile = base_profile.copy()
        modified_profile['overall_grade'] = grade
        
        print(f"\nTesting with overall grade: {grade}%")
        
        # Create a temporary profile with the modified overall grade
        temp_inputs = {
            'major_category': category_map.get(modified_profile.get('major'), modified_profile.get('major')),
            'course_category': 'Computer Science',  # Simplified for testing
            'overall_grade': grade,
            'faculty_grade': modified_profile.get('faculty_grade'),
            'prerequisite_grade': baseline_prereq
        }
        
        # Check the subject aptitude probabilities
        aptitude_probs = get_subject_aptitude_probs(temp_inputs)
        print(f"  Subject aptitude probs [L,M,H]: {np.round(aptitude_probs, 2)}")
        
        # Use a custom function to avoid prompting for grades
        result = run_with_prereq(baseline_prereq)  # Keep prereq grade constant
        
        # Update the result with the current profile values
        result['_profile'] = {'overall_grade': grade}
        
        sensitivity_results.append({
            'factor': 'Overall Grade',
            'value': grade, 
            'avg_gpa': result['avg_gpa'],
            'avg_letter': result['avg_letter'],
            'avg_percent': result['avg_percent'],
            'diff': result['avg_gpa'] - baseline_avg,
            'distribution': {k: v for k, v in result.items() if k in ['A', 'B', 'C', 'D', 'F']}
        })
    
    # ------------------------------------------------------------------------
    # 3. Faculty Grade Sensitivity
    # ------------------------------------------------------------------------
    print("\n=== FACULTY GRADE SENSITIVITY ===")
    faculty_grades = [70, 78, 85, 92]  # B-, B+, A-, A+
    
    for grade in faculty_grades:
        print(f"\nTesting with faculty grade: {grade}%")
        
        # Create a temporary profile with the modified faculty grade
        temp_inputs = {
            'major_category': category_map.get(base_profile.get('major'), base_profile.get('major')),
            'course_category': 'Computer Science',  # Simplified for testing
            'overall_grade': base_profile.get('overall_grade'),
            'faculty_grade': grade,
            'prerequisite_grade': baseline_prereq
        }
        
        # Check the subject aptitude probabilities
        aptitude_probs = get_subject_aptitude_probs(temp_inputs)
        print(f"  Subject aptitude probs [L,M,H]: {np.round(aptitude_probs, 2)}")
        
        # Use our custom function but keep prereq grade constant
        # We'll need to modify bayes_net_model inputs before running
        
        # Get course information
        course_row = courses_df[courses_df['course_code'] == course_code]
        class_size = int(course_row.iloc[0]['num_students'])
        professor_name = course_row.iloc[0]['prof']
        prof_row = prof_df[prof_df['name'] == professor_name]
        prof_rating = float(prof_row.iloc[0]['rating_val']) if not prof_row.empty else None
        prof_diff = float(prof_row.iloc[0]['diff_level']) if not prof_row.empty and pd.notna(prof_row.iloc[0]['diff_level']) else None
        
        # Create modified inputs
        model_inputs = {
            'major_category': category_map.get(base_profile.get('major'), base_profile.get('major')),
            'course_category': 'Computer Science',
            'overall_grade': base_profile.get('overall_grade'),
            'faculty_grade': grade,  # Use the current test value
            'prof_grade': None,
            'early_bird': base_profile.get('early_bird', False),
            'attendance_percent': base_profile.get('attendance_percent'),
            'class_time': base_profile.get('class_time'),
            'friends_in_class': base_profile.get('friends_in_class', False),
            'course_load': base_profile.get('course_load'),
            'job_status': base_profile.get('job_status'),
            'prerequisite_grade': baseline_prereq,
            'class_size': class_size,
            'course_code': course_code,
            'prof_rating': prof_rating,
            'prof_diff': prof_diff
        }
        
        # Run prediction with the modified inputs
        predictive = pyro.infer.Predictive(bayes_net_model, num_samples=num_samples)
        samples = predictive(model_inputs)
        
        # Calculate grade distribution
        grade_samples = samples["course_grade"].numpy()
        grade_dist = np.bincount(grade_samples, minlength=5) / num_samples
        
        # Grade letters and percentages
        grade_letters = ['A', 'B', 'C', 'D', 'F']
        grade_percentages = [87, 75, 65, 55, 40]
        
        # Create the distribution dictionary
        result = {
            grade_letters[i]: float(grade_dist[i]) for i in range(5)
        }
        
        # Calculate weighted average
        weighted_percent = sum(grade_dist[i] * grade_percentages[i] for i in range(5))
        final_letter, final_gpa = percent_to_letter_and_gpa(weighted_percent)
        
        result['avg_percent'] = weighted_percent
        result['avg_letter'] = final_letter
        result['avg_gpa'] = final_gpa
        
        # Print the result
        print(f"  Result: Avg GPA {final_gpa:.2f}, Grade dist: "
              f"A:{result['A']:.2f}, B:{result['B']:.2f}, C:{result['C']:.2f}, "
              f"D:{result['D']:.2f}, F:{result['F']:.2f}")
        
        sensitivity_results.append({
            'factor': 'Faculty Grade',
            'value': grade, 
            'avg_gpa': result['avg_gpa'],
            'avg_letter': result['avg_letter'],
            'avg_percent': result['avg_percent'],
            'diff': result['avg_gpa'] - baseline_avg,
            'distribution': {k: v for k, v in result.items() if k in ['A', 'B', 'C', 'D', 'F']}
        })
    
    # Print comprehensive sensitivity analysis results
    print("\n=============================================")
    print("SENSITIVITY ANALYSIS SUMMARY")
    print("=============================================")
    print(f"Baseline: {baseline_result['avg_letter']} ({baseline_result['avg_percent']:.1f}%, GPA: {baseline_avg:.2f})")
    
    # Group results by factor
    factor_groups = {}
    for result in sensitivity_results:
        factor = result['factor']
        if factor not in factor_groups:
            factor_groups[factor] = []
        factor_groups[factor].append(result)
    
    # Print results by factor
    for factor, results in factor_groups.items():
        print(f"\n{factor} Sensitivity:")
        print("-" * 70)
        print(f"{'Value':<12} {'Letter':<8} {'Percent':<10} {'GPA':<8} {'Change':<10} {'Grade Distribution'}")
        print("-" * 70)
        
        # Sort by value
        if all(isinstance(r['value'], (int, float)) for r in results):
            results = sorted(results, key=lambda x: x['value'])
        
        for result in results:
            dist_str = ", ".join([f"{g}: {p:.2f}" for g, p in result['distribution'].items()])
            print(f"{result['value']:<12} {result['avg_letter']:<8} {result['avg_percent']:.1f}%{' ':<6} {result['avg_gpa']:.2f}{' ':<4} {result['diff']:+.2f}{' ':<6} {dist_str}")
    
    # Find most influential factors
    factor_impacts = {}
    for factor, results in factor_groups.items():
        # Calculate the range of impact for this factor
        if len(results) > 1:
            avg_gpas = [r['avg_gpa'] for r in results]
            impact = max(avg_gpas) - min(avg_gpas)
            factor_impacts[factor] = impact
    
    # Sort factors by impact
    sorted_factors = sorted(factor_impacts.items(), key=lambda x: x[1], reverse=True)
    
    print("\n=============================================")
    print("FACTOR INFLUENCE RANKING")
    print("=============================================")
    print("Factors ranked by their influence on predicted grade:")
    for rank, (factor, impact) in enumerate(sorted_factors, 1):
        print(f"{rank}. {factor}: {impact:.2f} GPA points range")
    
    return sensitivity_results
# For running a more detailed analysis of how different factors affect grades
def run_full_analysis():
    """Run both the basic example and the full sensitivity analysis"""
    print("\n=== BASIC GRADE PREDICTION ===")
    example_usage()
    
    print("\n\n=== DETAILED SENSITIVITY ANALYSIS ===")
    print("(This may take a few minutes to complete)")
    improved_sensitivity_analysis()


run_full_analysis()


    
