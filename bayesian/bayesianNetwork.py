import re
import pyro
import pyro.distributions as dist
import torch
import numpy as np
import pandas as pd
import json

# Set random seed for reproducibility
pyro.set_rng_seed(0)

# Load data files
courses_df = pd.read_csv('csp/course_csp.csv')
prof_df = pd.read_csv('data/prof_qaulity_info.csv')
prof_df['diff_level'] = pd.to_numeric(prof_df['diff_level'], errors='coerce')

courses_full_df = pd.read_csv('bayesian/complete_courses.csv', usecols=['course_code', 'prereq_codes'])

# Load mapping from majors to broad categories
category_map = {}
with open('bayesian/category_map.txt') as f:
    for line in f:
        if ':' in line:
            major_name, category = line.strip().split(': ')
            category_map[major_name] = category

# Load major-course success matrix
with open('bayesian/major_course_success_matrix.json') as f:
    major_course_matrix = json.load(f)

# Department code to name mapping
dept_df = pd.read_csv('bayesian/complete_courses.csv', usecols=['department_name', 'department_code']).drop_duplicates()
dept_to_name = {code: name for code, name in zip(dept_df['department_code'], dept_df['department_name'])}

# -------------------------------------------------------------------------
# Utility Functions
# -------------------------------------------------------------------------

def get_prereqs(course_code):
    """Get prerequisites for a course"""
    normalized_code = course_code.replace(" ", "").upper()
    courses_full_df['normalized_code'] = courses_full_df['course_code'].str.replace(" ", "").str.upper()
    row = courses_full_df[courses_full_df['normalized_code'] == normalized_code]
    if row.empty:
        return []
    raw = row.iloc[0]['prereq_codes']
    if pd.isna(raw):
        return []
    return re.findall(r'([A-Z]{3,4}\s?\d{3})', raw)

def get_prof_of_course(code):
    """Get professor for a course"""
    row = courses_df[courses_df['course_code'] == code]
    return row.iloc[0]['prof'] if not row.empty else None

def letter_to_percent(letter):
    """Convert letter grade to percentage"""
    letter = letter.strip().upper()
    mapping = {
        'A+': 95, 'A': 87, 'A-': 82,
        'B+': 78, 'B': 75, 'B-': 71,
        'C+': 68, 'C': 65, 'C-': 61,
        'D+': 58, 'D': 55, 'D-': 51,
        'F': 40
    }
    
    if letter in mapping:
        return mapping[letter]
    else:
        # Handle numeric input
        try:
            numeric_grade = float(letter)
            if 0 <= numeric_grade <= 100:
                return numeric_grade
            else:
                return 50  # Default for invalid input
        except ValueError:
            return 50  # Default for invalid input

def percent_to_letter_and_gpa(percent):
    """Convert percentage to letter grade and GPA"""
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

def expand_grade_dist(basic_dist):
    """Expand basic A-F distribution to detailed A+, A, A-, etc."""
    expanded_dist = torch.zeros(13)
    
    # Distribution weights within each letter grade
    a_weights = torch.tensor([0.4, 0.35, 0.25])  # A+, A, A-
    b_weights = torch.tensor([0.3, 0.4, 0.3])    # B+, B, B-
    c_weights = torch.tensor([0.3, 0.4, 0.3])    # C+, C, C-
    d_weights = torch.tensor([0.3, 0.4, 0.3])    # D+, D, D-
    
    # Distribute probabilities
    expanded_dist[0:3] = basic_dist[0] * a_weights  # A+, A, A-
    expanded_dist[3:6] = basic_dist[1] * b_weights  # B+, B, B-
    expanded_dist[6:9] = basic_dist[2] * c_weights  # C+, C, C-
    expanded_dist[9:12] = basic_dist[3] * d_weights  # D+, D, D-
    expanded_dist[12] = basic_dist[4]  # F
    
    return expanded_dist

# -------------------------------------------------------------------------
# Factor Probability Functions
# -------------------------------------------------------------------------

def get_subject_aptitude_probs(inputs):
    base_vec = np.array([0.33, 0.34, 0.33])
    factors_applied = 0

    major_cat = inputs.get('major_category')
    course_cat = inputs.get('course_category')
    if major_cat and course_cat:
        factors_applied += 1
        key = f"{major_cat}|{course_cat}"
        if key in major_course_matrix:
            success_rates = major_course_matrix[key]
            major_vec = np.array([
                success_rates.get("D", 0) + success_rates.get("F", 0),
                success_rates.get("B", 0) + success_rates.get("C", 0),
                success_rates.get("A", 0)
            ])
            major_vec = major_vec / major_vec.sum()
            base_vec = 0.70 * base_vec + 0.30 * major_vec  # Down to 30% major alignment

    prof_grade = inputs.get('prof_grade')
    if prof_grade is not None:
        factors_applied += 1
        if prof_grade >= 85:
            prof_vec = np.array([0.1, 0.3, 0.6])
        elif prof_grade >= 75:
            prof_vec = np.array([0.2, 0.5, 0.3])
        elif prof_grade >= 65:
            prof_vec = np.array([0.4, 0.4, 0.2])
        else:
            prof_vec = np.array([0.6, 0.3, 0.1])
        base_vec = 0.4 * base_vec + 0.6 * prof_vec

    if factors_applied > 0:
        base_vec = base_vec / base_vec.sum()
    return base_vec.tolist()


def get_course_quality_probs(inputs):
    """Get probabilities for Course Quality based on professor rating and difficulty"""
    base_vec = np.array([0.33, 0.34, 0.33])  # Default [Low, Medium, High]
    factors_applied = 0
    
    # Factor 1: Rate My Professor score
    prof_rating = inputs.get('prof_rating')
    
    if prof_rating is not None:
        factors_applied += 1
        
        if prof_rating >= 4.0:
            rating_vec = np.array([0.1, 0.3, 0.6])  # High quality
        elif prof_rating >= 3.0:
            rating_vec = np.array([0.2, 0.6, 0.2])  # Medium quality
        else:
            rating_vec = np.array([0.6, 0.3, 0.1])  # Low quality
        
        base_vec = 0.4 * base_vec + 0.6 * rating_vec
    
    # Factor 2: Professor difficulty (inverted: high difficulty -> low quality)
    prof_diff = inputs.get('prof_diff')
    
    if prof_diff is not None:
        factors_applied += 1
        
        if prof_diff >= 4.0:
            diff_vec = np.array([0.6, 0.3, 0.1])  # Low quality (hard prof)
        elif prof_diff >= 3.0:
            diff_vec = np.array([0.2, 0.6, 0.2])  # Medium quality
        else:
            diff_vec = np.array([0.1, 0.3, 0.6])  # High quality (easy prof)
        
        base_vec = 0.3 * base_vec + 0.7 * diff_vec
    
    # Normalize if any factors were applied
    if factors_applied > 0:
        base_vec = base_vec / base_vec.sum()
    
    return base_vec.tolist()

def get_student_strength_probs(inputs):
    base_vec = np.array([0.33, 0.34, 0.33])
    factors_applied = 0

    overall_grade = inputs.get('overall_grade')
    if overall_grade is not None:
        factors_applied += 1
        if overall_grade >= 85:
            gpa_vec = np.array([0.05, 0.25, 0.70])
        elif overall_grade >= 75:
            gpa_vec = np.array([0.15, 0.60, 0.25])
        elif overall_grade >= 65:
            gpa_vec = np.array([0.40, 0.45, 0.15])
        else:
            gpa_vec = np.array([0.65, 0.30, 0.05])
        base_vec = 0.10 * base_vec + 0.90 * gpa_vec  # Now 90% GPA weight

    prereq_grade = inputs.get('prerequisite_grade')
    if prereq_grade is not None:
        factors_applied += 1
        if prereq_grade >= 85:
            prereq_vec = np.array([0.1, 0.2, 0.7])
        elif prereq_grade >= 75:
            prereq_vec = np.array([0.2, 0.5, 0.3])
        elif prereq_grade >= 65:
            prereq_vec = np.array([0.4, 0.5, 0.1])
        else:
            prereq_vec = np.array([0.7, 0.2, 0.1])
        base_vec = 0.90 * base_vec + 0.10 * prereq_vec

    if factors_applied > 0:
        base_vec = base_vec / base_vec.sum()
    return base_vec.tolist()

def get_participation_probs(inputs):
    base_vec = np.array([0.33, 0.34, 0.33])
    factors_applied = 0

    friends_in_class = inputs.get('friends_in_class')
    if friends_in_class is not None:
        factors_applied += 1
        friends_vec = np.array([0.1, 0.3, 0.6]) if friends_in_class else np.array([0.4, 0.4, 0.2])
        base_vec = 0.6 * base_vec + 0.4 * friends_vec

    course_load = inputs.get('course_load')
    if course_load is not None:
        factors_applied += 1
        if course_load > 5:
            load_vec = np.array([0.55, 0.3, 0.15])
        elif course_load == 5:
            load_vec = np.array([0.3, 0.4, 0.3])
        else:
            load_vec = np.array([0.1, 0.3, 0.6])
        base_vec = 0.5 * base_vec + 0.5 * load_vec

    early_bird = inputs.get('early_bird')
    class_times = inputs.get('class_time', [])
    if early_bird is not None and class_times:
        factors_applied += 1
        early_classes = sum(int(str(t).split(':')[0]) < 12 for t in class_times)
        early_ratio = early_classes / len(class_times)
        if early_bird and early_ratio > 0.5:
            time_vec = np.array([0.1, 0.3, 0.6])
        elif not early_bird and early_ratio > 0.5:
            time_vec = np.array([0.6, 0.3, 0.1])
        elif early_bird and early_ratio < 0.5:
            time_vec = np.array([0.3, 0.4, 0.3])
        else:
            time_vec = np.array([0.2, 0.3, 0.5])
        base_vec = 0.5 * base_vec + 0.5 * time_vec

    if factors_applied > 0:
        base_vec = base_vec / base_vec.sum()
    return base_vec.tolist()

# -------------------------------------------------------------------------
# Bayesian Network Model
# -------------------------------------------------------------------------

def bayes_net_model(inputs):
    p_apt = torch.tensor(get_subject_aptitude_probs(inputs), dtype=torch.float)
    subject_aptitude = pyro.sample("subject_aptitude", dist.Categorical(p_apt))

    p_quality = torch.tensor(get_course_quality_probs(inputs), dtype=torch.float)
    course_quality = pyro.sample("course_quality", dist.Categorical(p_quality))

    p_strength = torch.tensor(get_student_strength_probs(inputs), dtype=torch.float)
    student_strength = pyro.sample("student_strength", dist.Categorical(p_strength))

    p_participation = torch.tensor(get_participation_probs(inputs), dtype=torch.float)
    participation = pyro.sample("participation", dist.Categorical(p_participation))

    grade_cpt = torch.zeros((3, 3, 3, 3, 5))

    for a in range(3):
        for q in range(3):
            for s in range(3):
                for p in range(3):
                    grade_dist = [0.2, 0.2, 0.2, 0.2, 0.2]

                    if a == 2:
                        grade_dist = [0.35, 0.3, 0.2, 0.1, 0.05]
                    elif a == 1:
                        grade_dist = [0.2, 0.3, 0.3, 0.15, 0.05]
                    else:
                        grade_dist = [0.05, 0.15, 0.3, 0.3, 0.2]

                    if q == 2:
                        grade_dist = [grade_dist[0]*1.3, grade_dist[1]*1.2, grade_dist[2], grade_dist[3]*0.8, grade_dist[4]*0.7]
                    elif q == 0:
                        grade_dist = [grade_dist[0]*0.7, grade_dist[1]*0.8, grade_dist[2], grade_dist[3]*1.2, grade_dist[4]*1.3]

                    if s == 2:
                        grade_dist = [grade_dist[0]*1.3, grade_dist[1]*1.2, grade_dist[2], grade_dist[3]*0.8, grade_dist[4]*0.6]
                    elif s == 0:
                        grade_dist = [grade_dist[0]*0.6, grade_dist[1]*0.8, grade_dist[2], grade_dist[3]*1.2, grade_dist[4]*1.3]

                    if p == 2:
                        grade_dist = [grade_dist[0]*1.3, grade_dist[1]*1.1, grade_dist[2], grade_dist[3]*0.9, grade_dist[4]*0.7]
                    elif p == 0:
                        grade_dist = [grade_dist[0]*0.7, grade_dist[1]*0.9, grade_dist[2], grade_dist[3]*1.1, grade_dist[4]*1.3]

                    total = sum(grade_dist)
                    grade_dist = [d / total for d in grade_dist]
                    grade_cpt[a, q, s, p] = torch.tensor(grade_dist)

    grade_cpt[2, 2, 2, 2] = torch.tensor([0.7, 0.2, 0.07, 0.02, 0.01])
    grade_cpt[0, 0, 0, 0] = torch.tensor([0.01, 0.05, 0.14, 0.3, 0.5])

    basic_grade_probs = grade_cpt[int(subject_aptitude), int(course_quality), int(student_strength), int(participation)]
    basic_grade = pyro.sample("basic_grade", dist.Categorical(basic_grade_probs))

    detailed_grade_probs = expand_grade_dist(basic_grade_probs)
    final_grade = pyro.sample("final_grade", dist.Categorical(detailed_grade_probs))

    return final_grade


# -------------------------------------------------------------------------
# Inference and Prediction Functions
# -------------------------------------------------------------------------

def run_inference(course_code, user_profile, num_samples=1000):
    """Run inference on the Bayesian network to predict grade distribution"""
    # Get course information
    course_row = courses_df[courses_df['course_code'] == course_code]
    if course_row.empty:
        raise ValueError(f"Course code {course_code} not found.")
    
    class_size = int(course_row.iloc[0]['num_students'])
    professor_name = course_row.iloc[0]['prof']
    prof_row = prof_df[prof_df['name'] == professor_name]
    
    # Get professor rating and difficulty, allowing for overrides
    prof_rating = user_profile.get('prof_rating_override')
    if prof_rating is None:
        prof_rating = float(prof_row.iloc[0]['rating_val']) if not prof_row.empty else None
    
    prof_diff = user_profile.get('prof_diff_override')
    if prof_diff is None:
        prof_diff = float(prof_row.iloc[0]['diff_level']) if not prof_row.empty and pd.notna(prof_row.iloc[0]['diff_level']) else None
    
    # Handle prerequisite grades
    prereq_grades = {}
    avg_prereq_grade = user_profile.get('prerequisite_grade')
    
    if avg_prereq_grade is None:
        # Get prerequisites
        prereqs = get_prereqs(course_code)
        
        if 'prereq_grades' in user_profile:
            # Use already collected prereq grades
            prereq_grades = user_profile['prereq_grades']
            avg_prereq_grade = sum(prereq_grades.values()) / len(prereq_grades) if prereq_grades else None
        elif prereqs:
            # Ask user for prerequisite grades
            print("\nEnter your grades for the following prerequisites:")
            for c in prereqs:
                g = input(f"Grade for {c} (0-100 or letter): ").strip()
                prereq_grades[c] = letter_to_percent(g)
            
            # Calculate average
            avg_prereq_grade = sum(prereq_grades.values()) / len(prereq_grades) if prereq_grades else None
    
    # Get previous grades with same professor
    prof_grades = []
    for course, grade in prereq_grades.items() if prereq_grades else []:
        if get_prof_of_course(course) == professor_name:
            prof_grades.append(grade)
    
    avg_prof_grade = sum(prof_grades) / len(prof_grades) if prof_grades else None
    
    # Get major category and course category
    user_major = user_profile.get('major')
    major_category = category_map.get(user_major, user_major)
    
    dept_code_match = re.match(r'^[A-Za-z]+', course_code)
    course_category = 'Interdisciplinary'  # Default
    if dept_code_match:
        dept_code = dept_code_match.group(0)
        dept_name = dept_to_name.get(dept_code)
        if dept_name in category_map:
            course_category = category_map[dept_name]
    
    # Create model inputs
    inputs = {
        'major_category': major_category,
        'course_category': course_category,
        'overall_grade': user_profile.get('overall_grade'),
        'prof_grade': avg_prof_grade,
        'early_bird': user_profile.get('early_bird', False),
        'class_time': user_profile.get('class_time', []),
        'friends_in_class': user_profile.get('friends_in_class', False),
        'course_load': user_profile.get('course_load', 5),
        'prerequisite_grade': avg_prereq_grade,
        'class_size': class_size,
        'course_code': course_code,
        'prof_rating': prof_rating,
        'prof_diff': prof_diff
    }
    
    # Print input profile
    print("\nInput profile for grade prediction:")
    print(f"  Course: {course_code}")
    print(f"  Professor: {professor_name} (Rating: {prof_rating}, Difficulty: {prof_diff})")
    print(f"  Prerequisite avg grade: {avg_prereq_grade:.1f}%" if avg_prereq_grade is not None else "  No prerequisites")
    print(f"  Major category: {major_category}, Course category: {course_category}")
    print(f"  Overall grade: {user_profile.get('overall_grade')}")
    print(f"  Course load: {user_profile.get('course_load', 5)} courses")
    print(f"  Friends in class: {'Yes' if user_profile.get('friends_in_class', False) else 'No'}")
    print(f"  Morning person: {'Yes' if user_profile.get('early_bird', False) else 'No'}")
    
    # Run prediction
    predictive = pyro.infer.Predictive(bayes_net_model, num_samples=num_samples)
    samples = predictive(inputs)
    
    # Get samples for final grade
    final_grade_samples = samples["final_grade"].numpy()
    final_grade_dist = np.bincount(final_grade_samples, minlength=13) / num_samples
    
    # Grade letters and percentages
    grade_letters = ['A+', 'A', 'A-', 'B+', 'B', 'B-', 'C+', 'C', 'C-', 'D+', 'D', 'D-', 'F']
    grade_percentages = [95, 87, 82, 78, 75, 71, 68, 65, 61, 58, 55, 51, 40]
    
    # Create the distribution dictionary
    result = {
        grade_letters[i]: float(final_grade_dist[i]) for i in range(13)
    }
    
    # Calculate weighted average
    weighted_percent = sum(final_grade_dist[i] * grade_percentages[i] for i in range(13))
    final_letter, final_gpa = percent_to_letter_and_gpa(weighted_percent)
    
    # Add the average to the result
    result['avg_percent'] = weighted_percent
    result['avg_letter'] = final_letter
    result['avg_gpa'] = final_gpa
    
    return result

def run_grade_prediction():
    """Get user input and run the grade prediction"""
    course_code = input("\nEnter the course code (e.g., CISC352): ").strip().upper()
    
    # Collect user profile
    user_profile = {}
    user_profile['major'] = input("Enter your major: ").strip()
    
    try:
        user_profile['overall_grade'] = float(input("Enter your overall grade (0-100): ").strip())
    except ValueError:
        user_profile['overall_grade'] = 75  # Default to B
    
    try:
        user_profile['course_load'] = int(input("Enter your course load (number of courses this semester): ").strip())
    except ValueError:
        user_profile['course_load'] = 5  # Default
    
    friends_response = input("Do you have friends in this class? (y/n): ").strip().lower()
    user_profile['friends_in_class'] = friends_response.startswith('y')
    
    morning_response = input("Are you a morning person? (y/n): ").strip().lower()
    user_profile['early_bird'] = morning_response.startswith('y')
    
    class_times_input = input("Enter class times (comma-separated, e.g., 8:30,14:30,16:30): ").strip()
    if class_times_input:
        user_profile['class_time'] = class_times_input.split(',')
    else:
        user_profile['class_time'] = []
    
    prereqs = get_prereqs(course_code)
    prereq_grades = {}

    if prereqs:
        print("\nEnter your grades for the following prerequisites:")
        for c in prereqs:
            g = input(f"Grade for {c} (0-100 or letter): ").strip()
            prereq_grades[c] = letter_to_percent(g)

        avg_prereq_grade = sum(prereq_grades.values()) / len(prereq_grades)
        user_profile['prereq_grades'] = prereq_grades
        user_profile['prerequisite_grade'] = avg_prereq_grade
        
    # Run the prediction
    prediction = run_inference(course_code, user_profile)
    
    # Display results
    print("\n=============================================")
    print(f"GRADE PREDICTION FOR {course_code}")
    print("=============================================")
    
    # Display the grade distribution
    print("\nDetailed grade distribution:")
    
    # Group by letter grade categories
    a_grades = sum(prediction[g] for g in ['A+', 'A', 'A-'])
    b_grades = sum(prediction[g] for g in ['B+', 'B', 'B-'])
    c_grades = sum(prediction[g] for g in ['C+', 'C', 'C-'])
    d_grades = sum(prediction[g] for g in ['D+', 'D', 'D-'])
    f_grade = prediction['F']
    
    # Print both overview and detailed view
    print(f"A grades: {a_grades:.2f} (A+: {prediction['A+']:.2f}, A: {prediction['A']:.2f}, A-: {prediction['A-']:.2f})")
    print(f"B grades: {b_grades:.2f} (B+: {prediction['B+']:.2f}, B: {prediction['B']:.2f}, B-: {prediction['B-']:.2f})")
    print(f"C grades: {c_grades:.2f} (C+: {prediction['C+']:.2f}, C: {prediction['C']:.2f}, C-: {prediction['C-']:.2f})")
    print(f"D grades: {d_grades:.2f} (D+: {prediction['D+']:.2f}, D: {prediction['D']:.2f}, D-: {prediction['D-']:.2f})")
    print(f"F grade: {f_grade:.2f}")
    
    # Display the average results
    print(f"\nOverall predicted grade:")
    print(f"Average: {prediction['avg_percent']:.1f}% ({prediction['avg_letter']})")
    print(f"GPA equivalent: {prediction['avg_gpa']:.2f}")
    
    # Display interpretation
    letter_grade = prediction['avg_letter']
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
    
    return course_code, user_profile, prediction

# -------------------------------------------------------------------------
# Sensitivity Analysis
# -------------------------------------------------------------------------

def run_sensitivity_analysis(course_code, base_profile, num_samples=1000):
    """
    Run sensitivity analysis for different factors affecting the grade.
    Shows changes in detailed letter grades rather than just GPA.
    """
    print(f"\n=== RUNNING SENSITIVITY ANALYSIS FOR {course_code} ===")
    
    # Store the sensitivity results
    sensitivity_results = {}
    
    if 'prereq_grades' not in base_profile or 'prerequisite_grade' not in base_profile:
        prereqs = get_prereqs(course_code)
        prereq_grades = {}

        if prereqs:
            print("\nEnter your grades for the following prerequisites:")
            for c in prereqs:
                g = input(f"Grade for {c} (0-100 or letter): ").strip()
                prereq_grades[c] = letter_to_percent(g)

            # Calculate average
            avg_prereq_grade = sum(prereq_grades.values()) / len(prereq_grades) if prereq_grades else None
            base_profile['prereq_grades'] = prereq_grades
            base_profile['prerequisite_grade'] = avg_prereq_grade
        
    # Function to run inference with a modified profile
    def run_with_modified_profile(modified_profile, factor_name, factor_value):
        # Copy profile to avoid modifying the original
        test_profile = modified_profile.copy()
        
        # Use the prerequisite grades we already collected
        if 'prereq_grades' in base_profile:
            test_profile['prereq_grades'] = base_profile['prereq_grades']
            test_profile['prerequisite_grade'] = base_profile['prerequisite_grade']
        
        # Override the specific factor we're testing
        if factor_name == 'Prerequisite Grade':
            test_profile['prerequisite_grade'] = factor_value
        
        # Run inference
        result = run_inference(course_code, test_profile, num_samples)
        
        # Extract just the letter grade distribution
        grade_dist = {k: v for k, v in result.items() 
                      if k in ['A+', 'A', 'A-', 'B+', 'B', 'B-', 'C+', 'C', 'C-', 'D+', 'D', 'D-', 'F']}
        
        # Record the result
        if factor_name not in sensitivity_results:
            sensitivity_results[factor_name] = []
            
        sensitivity_results[factor_name].append({
            'value': factor_value, 
            'avg_letter': result['avg_letter'],
            'avg_percent': result['avg_percent'],
            'avg_gpa': result['avg_gpa'],
            'grade_distribution': grade_dist
        })
        
        return result
    
    # Baseline prediction
    print("\n=== BASELINE PREDICTION ===")
    baseline_result = run_inference(course_code, base_profile, num_samples)
    baseline_letter = baseline_result['avg_letter']
    baseline_percent = baseline_result['avg_percent']
    print(f"Baseline: {baseline_letter} ({baseline_percent:.1f}%, GPA: {baseline_result['avg_gpa']:.2f})")
    
    # Store baseline grade distribution for comparison
    baseline_dist = {k: v for k, v in baseline_result.items() 
                    if k in ['A+', 'A', 'A-', 'B+', 'B', 'B-', 'C+', 'C', 'C-', 'D+', 'D', 'D-', 'F']}
    
    # 1. Prerequisite Grade Sensitivity
    print("\n=== PREREQUISITE GRADE SENSITIVITY ===")
    prereq_grades = [50, 65, 75, 85, 95]  # Range from F to A+
    
    for grade in prereq_grades:
        print(f"Testing with prerequisite grade: {grade}%")
        run_with_modified_profile(base_profile, 'Prerequisite Grade', grade)
    
    # 2. Overall Grade Sensitivity
    print("\n=== OVERALL GRADE SENSITIVITY ===")
    overall_grades = [60, 70, 80, 90, 95]  # Range from D to A+
    
    for grade in overall_grades:
        print(f"Testing with overall grade: {grade}%")
        modified_profile = base_profile.copy()
        modified_profile['overall_grade'] = grade
        run_with_modified_profile(modified_profile, 'Overall Grade', grade)
    
    # 3. Course Load Sensitivity
    print("\n=== COURSE LOAD SENSITIVITY ===")
    course_loads = [3, 4, 5, 6, 7]  # Range from light to heavy
    
    for load in course_loads:
        print(f"Testing with course load: {load}")
        modified_profile = base_profile.copy()
        modified_profile['course_load'] = load
        run_with_modified_profile(modified_profile, 'Course Load', load)
    
    # 4. Friends in Class Sensitivity
    print("\n=== FRIENDS IN CLASS SENSITIVITY ===")
    
    for has_friends in [False, True]:
        print(f"Testing with friends in class: {has_friends}")
        modified_profile = base_profile.copy()
        modified_profile['friends_in_class'] = has_friends
        run_with_modified_profile(modified_profile, 'Friends in Class', has_friends)
    
    # 5. Morning/Night Person Sensitivity
    print("\n=== MORNING/NIGHT PERSON SENSITIVITY ===")
    
    for is_morning in [False, True]:
        print(f"Testing as {'morning' if is_morning else 'night'} person")
        modified_profile = base_profile.copy()
        modified_profile['early_bird'] = is_morning
        run_with_modified_profile(modified_profile, 'Morning Person', is_morning)
    
    # 6. Professor Rating Sensitivity
    print("\n=== PROFESSOR RATING SENSITIVITY ===")
    prof_ratings = [2.5, 3.0, 3.5, 4.0, 4.5]  # Range from poor to excellent
    
    for rating in prof_ratings:
        print(f"Testing with professor rating: {rating}/5.0")
        modified_profile = base_profile.copy()
        modified_profile['prof_rating_override'] = rating
        run_with_modified_profile(modified_profile, 'Professor Rating', rating)
    
    # 7. Professor Difficulty Sensitivity
    print("\n=== PROFESSOR DIFFICULTY SENSITIVITY ===")
    prof_difficulties = [1.5, 2.5, 3.5, 4.5]  # Range from easy to very difficult
    
    for difficulty in prof_difficulties:
        print(f"Testing with professor difficulty: {difficulty}/5.0")
        modified_profile = base_profile.copy()
        modified_profile['prof_diff_override'] = difficulty
        run_with_modified_profile(modified_profile, 'Professor Difficulty', difficulty)
    
    # 8. Class Time Sensitivity
    print("\n=== CLASS TIME SENSITIVITY ===")
    
    # Different class time scenarios
    class_time_scenarios = [
        (["8:30", "9:30", "10:30"], "All morning classes"),
        (["12:30", "13:30", "14:30"], "All afternoon classes"),
        (["8:30", "12:30", "16:30"], "Mixed classes")
    ]
    
    for times, description in class_time_scenarios:
        print(f"Testing with {description}")
        modified_profile = base_profile.copy()
        modified_profile['class_time'] = times
        run_with_modified_profile(modified_profile, 'Class Times', description)
    
    # Print comprehensive sensitivity analysis results
    print("\n=============================================")
    print("SENSITIVITY ANALYSIS SUMMARY")
    print("=============================================")
    
    # Define the order of grade letters for comparison
    grade_order = {
        'A+': 0, 'A': 1, 'A-': 2, 
        'B+': 3, 'B': 4, 'B-': 5, 
        'C+': 6, 'C': 7, 'C-': 8, 
        'D+': 9, 'D': 10, 'D-': 11, 
        'F': 12
    }
    
    # Get baseline grade index for comparison
    baseline_grade_idx = grade_order.get(baseline_letter, 6)  # Default to C if not found
    
    # Print results by factor
    for factor, results in sensitivity_results.items():
        print(f"\n{factor} Sensitivity:")
        print("-" * 90)
        print(f"{'Value':<12} {'Letter':<8} {'Percent':<10} {'GPA':<8} {'Change':<20} {'Top Grade Changes'}")
        print("-" * 90)
        
        # Sort by value if numeric
        if all(isinstance(r['value'], (int, float)) for r in results):
            results = sorted(results, key=lambda x: x['value'])
        
        for result in results:
            # Calculate the letter grade change
            result_grade_idx = grade_order.get(result['avg_letter'], 6)
            grade_diff = baseline_grade_idx - result_grade_idx  # Positive means improvement
            
            # Format the grade change
            if grade_diff > 0:
                grade_change = f"↑ {abs(grade_diff)} levels better"
            elif grade_diff < 0:
                grade_change = f"↓ {abs(grade_diff)} levels worse"
            else:
                grade_change = "No change"
            
            # Calculate key differences in grade distribution
            grade_diffs = {}
            for grade in grade_order.keys():
                baseline_prob = baseline_dist.get(grade, 0)
                result_prob = result['grade_distribution'].get(grade, 0)
                diff = result_prob - baseline_prob
                if abs(diff) >= 0.02:  # Only show significant changes (2% or more)
                    grade_diffs[grade] = diff
            
            # Sort and format top changes
            top_changes = sorted(grade_diffs.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
            top_changes_str = ", ".join([f"{g}: {d:+.1%}" for g, d in top_changes]) if top_changes else "No significant changes"
            
            # Print the formatted result
            value_str = str(result['value'])
            if isinstance(result['value'], bool):
                value_str = "Yes" if result['value'] else "No"
                
            print(f"{value_str:<12} {result['avg_letter']:<8} {result['avg_percent']:.1f}%{' ':<6} {result['avg_gpa']:.2f}{' ':<4} {grade_change:<20} {top_changes_str}")
    
    # Find most influential factors based on letter grade changes
    factor_influences = {}
    for factor, results in sensitivity_results.items():
        # Calculate the range of letter grade changes
        grade_indices = [grade_order.get(r['avg_letter'], 6) for r in results]
        if len(grade_indices) > 1:
            # Maximum difference in grade levels
            max_diff = max(grade_indices) - min(grade_indices)
            factor_influences[factor] = max_diff
            
            # Also look at percentage changes
            percent_values = [r['avg_percent'] for r in results]
            percent_range = max(percent_values) - min(percent_values)
            factor_influences[f"{factor} (percent)"] = percent_range
    
    # Sort factors by influence
    sorted_factors = sorted(factor_influences.items(), key=lambda x: x[1], reverse=True)
    
    print("\n=============================================")
    print("FACTOR INFLUENCE RANKING")
    print("=============================================")
    print("Factors ranked by their influence on predicted grade:")
    
    # Print grade level influence
    print("\nLetter Grade Level Changes:")
    grade_factors = [(f, v) for f, v in sorted_factors if not f.endswith("(percent)")]
    for rank, (factor, impact) in enumerate(grade_factors, 1):
        if impact > 0:
            print(f"{rank}. {factor}: {impact} grade levels")
        else:
            print(f"{rank}. {factor}: No effect on letter grade")
    
    # Print percentage influence
    print("\nPercentage Point Changes:")
    percent_factors = [(f.replace(" (percent)", ""), v) for f, v in sorted_factors if f.endswith("(percent)")]
    for rank, (factor, impact) in enumerate(percent_factors, 1):
        print(f"{rank}. {factor}: {impact:.1f} percentage points")
    
    return sensitivity_results


# -------------------------------------------------------------------------
# Main Function
# -------------------------------------------------------------------------

def main():
    print("\n=============================================")
    print("BAYESIAN GRADE PREDICTION SYSTEM")
    print("=============================================")

    course_code, user_profile, prediction = run_grade_prediction()

    # Ask for sensitivity
    response = input("\nRun simple sensitivity analysis on prereq grade? (y/n): ").strip().lower()
    if response.startswith('y'):
        run_sensitivity_analysis(course_code, user_profile)

if __name__ == "__main__":
    main()