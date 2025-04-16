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

def percent_to_letter_and_gpa(percent):
    """Convert percentage to letter grade and GPA with higher A+ threshold"""
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

def get_subject_aptitude_probs(inputs):
    base_vec = np.array([0.33, 0.34, 0.33])
    factors_applied = 0

    # Major-course alignment factor
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
            major_vec = major_vec / major_vec.sum() if major_vec.sum() > 0 else base_vec
            base_vec = 0.30 * base_vec + 0.70 * major_vec  # 70% impact from major-course match
    
    # Professor-specific performance factor 
    prof_grade = inputs.get('prof_grade')
    if prof_grade is not None:
        factors_applied += 1
        # Strong monotonic relationship
        if prof_grade >= 95:  # Nearly perfect with this professor
            prof_vec = np.array([0.00, 0.05, 0.95])  # Extremely high aptitude
        elif prof_grade >= 90:  # Excellent with this professor
            prof_vec = np.array([0.05, 0.10, 0.85])  # Very high aptitude
        elif prof_grade >= 85:  # Very good with this professor
            prof_vec = np.array([0.10, 0.20, 0.70])  # High aptitude
        elif prof_grade >= 80:  # Good with this professor
            prof_vec = np.array([0.15, 0.35, 0.50])  # Above average aptitude
        elif prof_grade >= 75:  # Average with this professor
            prof_vec = np.array([0.25, 0.60, 0.20])  # Average aptitude
        elif prof_grade >= 70:  # Below average with this professor
            prof_vec = np.array([0.40, 0.50, 0.10])  # Below average aptitude
        elif prof_grade >= 65:  # Poor with this professor
            prof_vec = np.array([0.60, 0.35, 0.05])  # Low aptitude
        else:  # Very poor with this professor
            prof_vec = np.array([0.80, 0.15, 0.05])  # Very low aptitude
        
        base_vec = 0.50 * base_vec + 0.50 * prof_vec  # 50% impact - significant

    # Normalize
    if factors_applied > 0:
        base_vec = base_vec / base_vec.sum()
    return base_vec.tolist()

def get_course_quality_probs(inputs):
    """Get probabilities for Course Quality based on professor rating and difficulty"""
    base_vec = np.array([0.33, 0.34, 0.33])  # Default [Low, Medium, High]
    factors_applied = 0
    
    # Factor 1: Rate My Professor score - neutral at 3.0, not 3.5
    prof_rating = inputs.get('prof_rating')
    
    if prof_rating is not None:
        factors_applied += 1
        
        if prof_rating >= 4.5:  # Excellent professor
            rating_vec = np.array([0.05, 0.15, 0.80])  # Very high quality
        elif prof_rating >= 4.0:  # Very good professor
            rating_vec = np.array([0.10, 0.20, 0.70])  # High quality
        elif prof_rating >= 3.5:  # Good professor
            rating_vec = np.array([0.15, 0.25, 0.60])  # Above average quality
        elif prof_rating >= 3.0:  # Average professor
            rating_vec = np.array([0.25, 0.50, 0.25])  # Neutral/average quality
        elif prof_rating >= 2.5:  # Below average professor
            rating_vec = np.array([0.45, 0.35, 0.20])  # Below average quality
        elif prof_rating >= 1.5:  # Below average professor
            rating_vec = np.array([0.7, 0.2, 0.10])  # Below average quality
        else:  # Poor professor
            rating_vec = np.array([0.80, 0.15, 0.05])  # Low quality
        
        base_vec = 0.30 * base_vec + 0.70 * rating_vec  # 70% impact from rating
    
    # Factor 2: Professor difficulty (inverted: high difficulty -> low quality)
    prof_diff = inputs.get('prof_diff')
    
    if prof_diff is not None:
        factors_applied += 1
        
        if prof_diff >= 4.5:  # Extremely difficult
            diff_vec = np.array([0.85, 0.1, 0.05])  # Very low quality
        elif prof_diff >= 4.0:  # Very difficult
            diff_vec = np.array([0.60, 0.30, 0.10])  # Low quality
        elif prof_diff >= 3.5:  # Difficult
            diff_vec = np.array([0.45, 0.40, 0.15])  # Below average quality
        elif prof_diff >= 3.0:  # Average/moderate difficulty
            diff_vec = np.array([0.25, 0.50, 0.25])  # Neutral/average quality
        elif prof_diff >= 2.5:  # Somewhat easy
            diff_vec = np.array([0.15, 0.35, 0.50])  # Above average quality
        elif prof_diff >= 2.0:  # Easy
            diff_vec = np.array([0.10, 0.20, 0.60])  # High quality
        elif prof_diff >= 1.0:  # Easy
            diff_vec = np.array([0.10, 0.20, 0.70])  # High quality
        else:  # Very easy
            diff_vec = np.array([0.05, 0.1, 0.85])  # Very high quality
        
        base_vec = 0.40 * base_vec + 0.60 * diff_vec  # 60% impact from difficulty
    
    # Normalize if any factors were applied
    if factors_applied > 0:
        base_vec = base_vec / base_vec.sum()
    
    return base_vec.tolist()

def get_student_strength_probs(inputs):
    base_vec = np.array([0.33, 0.34, 0.33])  # Neutral default
    factors_applied = 0
    
    # Overall Grade Impact
    overall_grade = inputs.get('overall_grade')
    if overall_grade is not None:
        factors_applied += 1
        if overall_grade >= 97:  # Near perfect
            gpa_vec = np.array([0.00, 0.00, 1.00])  # Guaranteed high strength
        elif overall_grade >= 95:  # Exceptional
            gpa_vec = np.array([0.00, 0.01, 0.99])
        elif overall_grade >= 90:  # Excellent
            gpa_vec = np.array([0.01, 0.02, 0.97])
        elif overall_grade >= 85:  # Very good
            gpa_vec = np.array([0.02, 0.08, 0.90])
        elif overall_grade >= 80:  # Good
            gpa_vec = np.array([0.05, 0.20, 0.75])
        elif overall_grade >= 75:  # Above average
            gpa_vec = np.array([0.15, 0.55, 0.30])
        elif overall_grade >= 70:  # Average
            gpa_vec = np.array([0.30, 0.60, 0.10])  
        elif overall_grade >= 65:  # Below average
            gpa_vec = np.array([0.60, 0.35, 0.05])  
        elif overall_grade >= 60:  # Poor
            gpa_vec = np.array([0.80, 0.18, 0.02])  
        elif overall_grade >= 50:  # Failing
            gpa_vec = np.array([0.92, 0.07, 0.01])  
        elif overall_grade >= 40:  # Severely failing
            gpa_vec = np.array([0.95, 0.04, 0.01])
        elif overall_grade >= 30:  # Very poor
            gpa_vec = np.array([0.97, 0.03, 0.00])
        else:  # Extremely poor
            gpa_vec = np.array([0.99, 0.01, 0.00])
        
        # Overall grade has 95% weight
        base_vec = 0.05 * base_vec + 0.95 * gpa_vec

    # Prerequisite Impact
    prereq_grade = inputs.get('prerequisite_grade')
    if prereq_grade is not None:
        factors_applied += 1
        # Realistic progression across the prerequisite grade spectrum
        if prereq_grade >= 97:  # Near perfect prerequisites
            prereq_vec = np.array([0.00, 0.00, 1.00])
        elif prereq_grade >= 95:
            prereq_vec = np.array([0.00, 0.01, 0.99])
        elif prereq_grade >= 90:
            prereq_vec = np.array([0.01, 0.04, 0.95])
        elif prereq_grade >= 85:
            prereq_vec = np.array([0.02, 0.08, 0.90])
        elif prereq_grade >= 80:
            prereq_vec = np.array([0.05, 0.15, 0.80])
        elif prereq_grade >= 75:
            prereq_vec = np.array([0.10, 0.30, 0.60])
        elif prereq_grade >= 70:
            prereq_vec = np.array([0.30, 0.60, 0.10]) 
        elif prereq_grade >= 65:
            prereq_vec = np.array([0.60, 0.35, 0.05])  
        elif prereq_grade >= 60:
            prereq_vec = np.array([0.80, 0.18, 0.02])  
        elif prereq_grade >= 50:
            prereq_vec = np.array([0.94, 0.05, 0.01])  
        elif prereq_grade >= 40:  # Severely failing
            prereq_vec = np.array([0.96, 0.03, 0.01])
        elif prereq_grade >= 30:  # Very poor
            prereq_vec = np.array([0.98, 0.02, 0.00])
        else:  # Extremely poor
            prereq_vec = np.array([0.99, 0.01, 0.00])

        if overall_grade is not None:
            # With overall grade, prereqs have 60% weight 
            base_vec = 0.40 * base_vec + 0.60 * prereq_vec
        else:
            # Without overall grade, prereqs have 95% weight
            base_vec = 0.05 * base_vec + 0.95 * prereq_vec
    
    # Course Load Impact
    course_load = inputs.get('course_load')
    if course_load is not None:
        factors_applied += 1
        # Strictly monotonic impact based on course load
        if course_load == 3:  # Optimal light load
            load_vec = np.array([0.05, 0.15, 0.80])  # Very positive impact
        elif course_load == 4:  # Good balanced load
            load_vec = np.array([0.08, 0.22, 0.70])  # Positive impact
        elif course_load == 5:  # Standard/neutral load
            load_vec = np.array([0.33, 0.34, 0.33])  # Neutral impact
        elif course_load == 6:  # Heavy load
            load_vec = np.array([0.35, 0.45, 0.20])  # Negative impact
        else:  # Very heavy load (7+)
            load_vec = np.array([0.60, 0.30, 0.10])  # Strong negative impact
        
        # Apply course load with 15% weight
        base_vec = 0.85 * base_vec + 0.15 * load_vec
    
    # Special cases section
    
    # 1. Special case for exceptional students with excellent prerequisites
    if overall_grade is not None and prereq_grade is not None:
        if overall_grade >= 97 and prereq_grade >= 97:
            # Absolutely perfect student - guaranteed high strength
            base_vec = np.array([0.00, 0.00, 1.00])
        elif overall_grade >= 95 and prereq_grade >= 95:
            # Nearly perfect student
            base_vec = np.array([0.00, 0.01, 0.99])
        elif overall_grade >= 90 and prereq_grade >= 90:
            # Excellent student
            base_vec = np.array([0.01, 0.02, 0.97])
    
    # 2. Special case for struggling students to ensure realistic outcomes
    if overall_grade is not None and overall_grade < 60:
        # Student with failing overall GPA - cap high strength probability
        # based on a realistic assessment of their likelihood to excel
        
        # Calculate tentative high strength probability
        high_strength_prob = base_vec[2]
        
        # Apply realistic caps based on overall grade
        if overall_grade < 30:  # Extremely poor
            # Cap high strength at 0% maximum
            high_strength_prob = 0.0
            # Ensure medium strength is also severely limited
            medium_strength_prob = min(base_vec[1], 0.02)
            # Low strength dominates
            low_strength_prob = 1.0 - high_strength_prob - medium_strength_prob
            base_vec = np.array([low_strength_prob, medium_strength_prob, high_strength_prob])
        elif overall_grade < 40:  # Very poor
            # Cap high strength at 0.5% maximum
            high_strength_prob = min(high_strength_prob, 0.005)
            # Ensure medium strength is also severely limited
            medium_strength_prob = min(base_vec[1], 0.05)
            # Low strength dominates
            low_strength_prob = 1.0 - high_strength_prob - medium_strength_prob
            base_vec = np.array([low_strength_prob, medium_strength_prob, high_strength_prob])
        elif overall_grade < 50:  # failing
            # Cap high strength at 1% maximum
            high_strength_prob = min(high_strength_prob, 0.01)
            # Ensure medium strength is also limited
            medium_strength_prob = min(base_vec[1], 0.09)
            # Low strength dominates
            low_strength_prob = 1.0 - high_strength_prob - medium_strength_prob
            base_vec = np.array([low_strength_prob, medium_strength_prob, high_strength_prob])
        elif overall_grade < 55:  # Failing
            # Cap high strength at 2% maximum
            high_strength_prob = min(high_strength_prob, 0.02)
            # Ensure medium strength is also appropriately limited
            medium_strength_prob = min(base_vec[1], 0.15)
            # Low strength dominates
            low_strength_prob = 1.0 - high_strength_prob - medium_strength_prob
            base_vec = np.array([low_strength_prob, medium_strength_prob, high_strength_prob])
        else:  # Borderline failing
            # Cap high strength at 3% maximum
            high_strength_prob = min(high_strength_prob, 0.03)
            # Medium strength also limited but less severely
            medium_strength_prob = min(base_vec[1], 0.25)
            # Low strength still dominates
            low_strength_prob = 1.0 - high_strength_prob - medium_strength_prob
            base_vec = np.array([low_strength_prob, medium_strength_prob, high_strength_prob])
    
    # 3. Special case for multiple negative factors
    # When multiple negative factors align, the effect should be multiplicative
    negative_factors = 0
    
    # Count negative factors
    if overall_grade is not None and overall_grade < 65:
        negative_factors += 1
    if prereq_grade is not None and prereq_grade < 65:
        negative_factors += 1
    if course_load is not None and course_load > 5:
        negative_factors += 1
    
    # Apply multiplicative penalty for multiple negative factors
    if negative_factors >= 3:
        # With 3+ negative factors, drastically reduce high strength probability
        reduction_factor = 0.5 ** (negative_factors - 2)  # Exponential reduction
        
        # Apply reduction to high strength probability
        high_strength = base_vec[2] * reduction_factor
        
        # Redistribute to low strength primarily
        low_strength = base_vec[0] + base_vec[2] * (1 - reduction_factor) * 0.8
        medium_strength = base_vec[1] + base_vec[2] * (1 - reduction_factor) * 0.2
        
        base_vec = np.array([low_strength, medium_strength, high_strength])
    
    # Ensure proper normalization
    if factors_applied > 0:
        base_vec = base_vec / base_vec.sum()
    
    return base_vec.tolist()


def get_participation_probs(inputs):
    base_vec = np.array([0.33, 0.34, 0.33])
    factors_applied = 0

    # Friends in class effect
    friends_in_class = inputs.get('friends_in_class')
    if friends_in_class is not None:
        factors_applied += 1
        friends_vec = np.array([0.10, 0.30, 0.60]) if friends_in_class else np.array([0.40, 0.30, 0.30])
        base_vec = 0.50 * base_vec + 0.50 * friends_vec  # 50% impact

       # Morning/Evening person match with class time
    early_bird = inputs.get('early_bird')
    class_times = inputs.get('class_time', [])
    if early_bird is not None and class_times:
        factors_applied += 1
        # Calculate proportion of morning classes
        early_classes = sum(int(str(t).split(':')[0]) < 12 for t in class_times)
        early_ratio = early_classes / len(class_times) if len(class_times) > 0 else 0
        
        # Match or mismatch effect
        if early_bird and early_ratio >= 0.5:  # Morning person with morning classes
            time_vec = np.array([0.10, 0.20, 0.70])  # Positive effect
        elif not early_bird and early_ratio < 0.5:  # Night person with evening classes
            time_vec = np.array([0.10, 0.20, 0.70])  # Positive effect
        elif early_bird and early_ratio < 0.5:  # Morning person with evening classes
            time_vec = np.array([0.33, 0.34, 0.33])  # neutral effect
        elif not early_bird and early_ratio >= 0.5:  # Night person with morning classes
            time_vec = np.array([0.50, 0.30, 0.20])  # Significant negative effect
        else:  # Mixed schedule - neutral effect
            time_vec = np.array([0.33, 0.34, 0.33])  # Neutral effect
   
        base_vec = 0.95 * base_vec + 0.05 * time_vec

    if factors_applied > 0:
        base_vec = base_vec / base_vec.sum()
    return base_vec.tolist()

def expand_grade_dist_adaptive(basic_dist, inputs):
    """
    Expand basic A-F distribution to detailed A+, A, A- etc. 
    Adapts based on student profile for more realistic distributions.
    """
    expanded_dist = torch.zeros(13)
    
    # Check overall student profile to determine if A+ should be heavily restricted
    overall_grade = inputs.get('overall_grade')
    prereq_grade = inputs.get('prerequisite_grade')
    
    # Identify if this is a struggling student
    is_struggling = (overall_grade is not None and overall_grade < 65) or \
                    (prereq_grade is not None and prereq_grade < 65)
    
    # Identify if this is a very poor student
    is_very_poor = (overall_grade is not None and overall_grade < 40) or \
                   (prereq_grade is not None and prereq_grade < 40)
    
    # For struggling students, use a different distribution for A grades
    if is_struggling:
        # For struggling students, A+ should be much less common even when they get an A
        struggling_a_weights = torch.tensor([0.15, 0.35, 0.50])  # Even less A+
        struggling_b_weights = torch.tensor([0.20, 0.40, 0.40])  # Less B+
        
        # Extreme A+ bias for exceptional students
        if basic_dist[0] > 0.98:  # Perfect student
            a_weights = torch.tensor([0.90, 0.08, 0.02])
        elif basic_dist[0] > 0.95:  # Nearly perfect
            a_weights = torch.tensor([0.80, 0.15, 0.05])
        elif basic_dist[0] > 0.90:  # Exceptional
            a_weights = torch.tensor([0.70, 0.20, 0.10])
        elif basic_dist[0] > 0.80:  # Excellent
            a_weights = torch.tensor([0.60, 0.25, 0.15])
        elif basic_dist[0] > 0.60:  # Very good
            a_weights = torch.tensor([0.40, 0.30, 0.30])
        else:  # Average or below
            a_weights = struggling_a_weights  # Use struggling student distribution
        
        # Use modified B weights for struggling students
        b_weights = struggling_b_weights
    else:
        # For good students, use the original distribution with strong A+ bias
        if basic_dist[0] > 0.98:  # Perfect student
            a_weights = torch.tensor([0.99, 0.01, 0.00]) 
        elif basic_dist[0] > 0.95:  # Nearly perfect
            a_weights = torch.tensor([0.97, 0.02, 0.01])  
        elif basic_dist[0] > 0.90:  # Exceptional
            a_weights = torch.tensor([0.95, 0.04, 0.01]) 
        elif basic_dist[0] > 0.80:  # Excellent
            a_weights = torch.tensor([0.85, 0.10, 0.05])  
        elif basic_dist[0] > 0.70:  # Very good
            a_weights = torch.tensor([0.75, 0.15, 0.10])  
        elif basic_dist[0] > 0.60:  # Good
            a_weights = torch.tensor([0.65, 0.20, 0.15])  
        elif basic_dist[0] > 0.40:  # Above average
            a_weights = torch.tensor([0.55, 0.25, 0.20])  
        elif basic_dist[0] > 0.20:  # Average 
            a_weights = torch.tensor([0.45, 0.30, 0.25])  
        else:  # Below average
            a_weights = torch.tensor([0.35, 0.35, 0.30]) 
        
        # Standard B weights
        b_weights = torch.tensor([0.40, 0.40, 0.20])    # B+, B, B-
    
    # Standard distributions for C and D grades
    c_weights = torch.tensor([0.35, 0.40, 0.25])    # C+, C, C-
    d_weights = torch.tensor([0.35, 0.40, 0.25])    # D+, D, D-
    
    # Distribute probabilities
    expanded_dist[0:3] = basic_dist[0] * a_weights  # A+, A, A-
    expanded_dist[3:6] = basic_dist[1] * b_weights  # B+, B, B-
    expanded_dist[6:9] = basic_dist[2] * c_weights  # C+, C, C-
    expanded_dist[9:12] = basic_dist[3] * d_weights  # D+, D, D-
    expanded_dist[12] = basic_dist[4]  # F
    
    # Special handling for very poor students - increase F probability
    if is_very_poor:
        # For very poor students, ensure F probability is higher
        if overall_grade is not None:
            if overall_grade < 30:
                # Below 30% should be almost guaranteed F
                expanded_dist[12] = max(expanded_dist[12], 0.85)
            elif overall_grade < 40:
                # Below 40% should have very high F probability
                expanded_dist[12] = max(expanded_dist[12], 0.70)
                
        # Normalize after adjustment
        if expanded_dist[12] > basic_dist[4]:
            # Reduce other probabilities proportionally
            total_others = 1.0 - expanded_dist[12]
            if sum(expanded_dist[0:12]) > 0:  # Avoid division by zero
                factor = total_others / sum(expanded_dist[0:12])
                expanded_dist[0:12] = expanded_dist[0:12] * factor
    
    return expanded_dist

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

    # fill all combinations with a reasonable fallback distribution
    for a in range(3):
        for q in range(3):
            for s in range(3):
                for p in range(3):
                    # Default grade distribution 
                    grade_dist = [0.15, 0.20, 0.30, 0.25, 0.10]
                    grade_cpt[a, q, s, p] = torch.tensor(grade_dist)
    
    # Strong student strength factor
        for q in range(3):
            for p in range(3):
                grade_cpt[a, q, 0, p] = torch.tensor([0.00, 0.05, 0.15, 0.35, 0.45])
    
    # MEDIUM student strength 
    for a in range(3):
        for q in range(3):
            for p in range(3):
                grade_cpt[a, q, 1, p] = torch.tensor([0.10, 0.30, 0.40, 0.15, 0.05])
    
    # HIGH student strength
    for a in range(3):
        for q in range(3):
            for p in range(3):
                grade_cpt[a, q, 2, p] = torch.tensor([0.85, 0.12, 0.02, 0.01, 0.00])
    
    # Now apply modifiers for other factors
    
    # Course quality modifiers 
    for a in range(3):
        for p in range(3):
            # LOW quality reduces grade outcomes
            grade_cpt[a, 0, 2, p] = torch.tensor([0.70, 0.20, 0.08, 0.02, 0.00])
            grade_cpt[a, 0, 1, p] = torch.tensor([0.05, 0.20, 0.40, 0.25, 0.10]) 
            grade_cpt[a, 0, 0, p] = torch.tensor([0.00, 0.02, 0.13, 0.35, 0.50])
            
            # HIGH quality improves grade outcomes
            grade_cpt[a, 2, 2, p] = torch.tensor([0.90, 0.08, 0.01, 0.01, 0.00])
            grade_cpt[a, 2, 1, p] = torch.tensor([0.20, 0.40, 0.30, 0.08, 0.02])
            grade_cpt[a, 2, 0, p] = torch.tensor([0.03, 0.10, 0.20, 0.32, 0.35])
    
    # Subject aptitude modifiers 
    for q in range(3):
        for p in range(3):
            # LOW aptitude reduces grade outcomes
            grade_cpt[0, q, 2, p] = torch.tensor([0.70, 0.20, 0.08, 0.02, 0.00])
            grade_cpt[0, q, 1, p] = torch.tensor([0.05, 0.20, 0.40, 0.25, 0.10])
            grade_cpt[0, q, 0, p] = torch.tensor([0.00, 0.02, 0.13, 0.35, 0.50])
            
            # High student strength + medium aptitude + high quality
        grade_cpt[1, 2, 2, 2] = torch.tensor([0.95, 0.04, 0.01, 0.00, 0.00])
        grade_cpt[1, 2, 2, 1] = torch.tensor([0.92, 0.06, 0.02, 0.00, 0.00])
        grade_cpt[1, 2, 2, 0] = torch.tensor([0.90, 0.08, 0.02, 0.00, 0.00])
        
        # WORST CASE: All factors LOW - realistic for failing students
        grade_cpt[0, 0, 0, 0] = torch.tensor([0.00, 0.00, 0.05, 0.25, 0.70])
        
        # OTHER POOR COMBINATIONS
        # Low strength with other factors also poor
        grade_cpt[0, 0, 0, 1] = torch.tensor([0.00, 0.01, 0.09, 0.30, 0.60])
        grade_cpt[0, 1, 0, 0] = torch.tensor([0.00, 0.01, 0.09, 0.30, 0.60])
        grade_cpt[1, 0, 0, 0] = torch.tensor([0.00, 0.01, 0.09, 0.30, 0.60])
        
        # Low strength but one good factor
        grade_cpt[2, 0, 0, 0] = torch.tensor([0.00, 0.03, 0.12, 0.35, 0.50])
        grade_cpt[0, 2, 0, 0] = torch.tensor([0.00, 0.03, 0.12, 0.35, 0.50])
        grade_cpt[0, 0, 0, 2] = torch.tensor([0.00, 0.03, 0.12, 0.35, 0.50])

        # Sample from the CPT based on the sampled factor states
        basic_grade_probs = grade_cpt[int(subject_aptitude), int(course_quality), int(student_strength), int(participation)]
        basic_grade = pyro.sample("basic_grade", dist.Categorical(basic_grade_probs))

        # Modified expansion that adapts to student profile
        detailed_grade_probs = expand_grade_dist_adaptive(basic_grade_probs, inputs)
        final_grade = pyro.sample("final_grade", dist.Categorical(detailed_grade_probs))

        return final_grade

def run_inference(course_code, user_profile, num_samples=10000):
    try:
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
            prof_rating = float(prof_row.iloc[0]['rating_val']) if not prof_row.empty else 3.0  # Default to neutral
        
        prof_diff = user_profile.get('prof_diff_override')
        if prof_diff is None:
            prof_diff = float(prof_row.iloc[0]['diff_level']) if not prof_row.empty and pd.notna(prof_row.iloc[0]['diff_level']) else 3.0  # Default to neutral
        
        # Handle prerequisite grades with better error handling
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
                    # Try to convert to float first (numerical grade)
                    prereq_grades[c] = float(g)

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
        
        predictive = pyro.infer.Predictive(bayes_net_model, num_samples=num_samples)
        samples = predictive(inputs)
        
        # Get samples for final grade
        final_grade_samples = samples["final_grade"].numpy()
        final_grade_dist = np.bincount(final_grade_samples, minlength=13) / num_samples
        
        # Grade letters and percentages 
        grade_letters = ['A+', 'A', 'A-', 'B+', 'B', 'B-', 'C+', 'C', 'C-', 'D+', 'D', 'D-', 'F']
        grade_percentages = [90, 85, 80, 77, 73, 70, 67, 63, 60, 57, 53, 50, 25]
        
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
    
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        # Return a default result in case of error
        default_result = {
            'A+': 0.0, 'A': 0.0, 'A-': 0.0, 
            'B+': 0.0, 'B': 0.0, 'B-': 0.0,
            'C+': 0.0, 'C': 1.0, 'C-': 0.0,  # Default to C
            'D+': 0.0, 'D': 0.0, 'D-': 0.0,
            'F': 0.0,
            'avg_percent': 65.0,
            'avg_letter': 'C',
            'avg_gpa': 2.0
        }
        return default_result

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
            g = input(f"Grade for {c} (0-100): ").strip()
            prereq_grades[c] = float(g)

        avg_prereq_grade = sum(prereq_grades.values()) / len(prereq_grades)
        user_profile['prereq_grades'] = prereq_grades
        user_profile['prerequisite_grade'] = avg_prereq_grade

    # Run the prediction
    prediction = run_inference(course_code, user_profile)
    
    ## Display results
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
    if prediction['avg_letter'] == 'F':
        print(f"Average: <50% (F)")
    else:
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

# Sensitivity Analysis

def run_sensitivity_analysis(course_code, base_profile, num_samples=2000):
    """
    Run sensitivity analysis for different factors affecting the grade.
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
                g = input(f"Grade for {c} (0-100): ").strip()
                prereq_grades[c] = float(g)

            # Calculate average
            avg_prereq_grade = sum(prereq_grades.values()) / len(prereq_grades) if prereq_grades else None
            base_profile['prereq_grades'] = prereq_grades
            base_profile['prerequisite_grade'] = avg_prereq_grade
        
    # Function to run inference with a modified profile
    def run_with_modified_profile(modified_profile, factor_name, factor_value):
        # Deep copy profile to avoid modifying the original
        test_profile = modified_profile.copy()

        if factor_name == 'Prerequisite Grade':
            test_profile['prerequisite_grade'] = factor_value
            # Create a new prereq_grades dict that matches the desired average
            if 'prereq_grades' in test_profile and test_profile['prereq_grades']:
                orig_grades = test_profile['prereq_grades']
                orig_avg = sum(orig_grades.values()) / len(orig_grades)
                
                # Instead of just shifting by the difference, rescale proportionally
                # This ensures no grade goes above 100 or below 0
                if orig_avg > 0:  # Avoid division by zero
                    scale_factor = factor_value / orig_avg
                    new_grades = {}
                    for course, grade in orig_grades.items():
                        # Scale each grade, but cap at 100 and floor at min(original, 25)
                        new_grade = min(100, grade * scale_factor)
                        # Don't let grades drop below the original or 25
                        new_grade = max(min(grade, 25), new_grade)
                        new_grades[course] = new_grade
                    
                    # Recalculate the actual average after capping
                    actual_avg = sum(new_grades.values()) / len(new_grades)
                    
                    # Additional adjustment if needed to hit target more precisely
                    if abs(actual_avg - factor_value) > 1.0:
                        adjustment = factor_value - actual_avg
                        # Apply small adjustment while respecting bounds
                        for course in new_grades:
                            if adjustment > 0 and new_grades[course] < 100:
                                new_grades[course] = min(100, new_grades[course] + adjustment)
                            elif adjustment < 0 and new_grades[course] > 50:
                                new_grades[course] = max(50, new_grades[course] + adjustment)
                    
                    test_profile['prereq_grades'] = new_grades
                    
                    # Update prof_grade if needed based on new prereq grades
                    professor_name = courses_df[courses_df['course_code'] == course_code].iloc[0]['prof']
                    prof_grades = []
                    for course, grade in new_grades.items():
                        if get_prof_of_course(course) == professor_name:
                            prof_grades.append(grade)
                    
                    if prof_grades:
                        test_profile['prof_grade'] = sum(prof_grades) / len(prof_grades)
        
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
    baseline_result = run_inference(course_code, base_profile, num_samples*2)
    baseline_letter = baseline_result['avg_letter']
    baseline_percent = baseline_result['avg_percent']
    print(f"Baseline: {baseline_letter} ({baseline_percent:.1f}%, GPA: {baseline_result['avg_gpa']:.2f})")
    
    # Store baseline grade distribution for comparison
    baseline_dist = {k: v for k, v in baseline_result.items() 
                    if k in ['A+', 'A', 'A-', 'B+', 'B', 'B-', 'C+', 'C', 'C-', 'D+', 'D', 'D-', 'F']}
    
    # 1. Prerequisite Grade Sensitivity 
    print("\n=== PREREQUISITE GRADE SENSITIVITY ===")
    prereq_grades = [25, 50, 65, 75, 85, 90, 95, 100] 
    
    for grade in prereq_grades:
        print(f"Testing with prerequisite grade: {grade}%")
        run_with_modified_profile(base_profile, 'Prerequisite Grade', grade)
    
    # 2. Overall Grade Sensitivity 
    print("\n=== OVERALL GRADE SENSITIVITY ===")
    overall_grades = [25, 50, 65, 75, 85, 90, 95, 100]  
    
    for grade in overall_grades:
        print(f"Testing with overall grade: {grade}%")
        modified_profile = base_profile.copy()
        modified_profile['overall_grade'] = grade
        run_with_modified_profile(modified_profile, 'Overall Grade', grade)
    
    # 3. Course Load Sensitivity
    print("\n=== COURSE LOAD SENSITIVITY ===")
    course_loads = [3, 4, 5, 6, 7]  
    
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
    prof_ratings = [0.5, 1.5, 2.5, 3.0, 3.5, 4.0, 4.5]  # Range from poor to excellent
    
    for rating in prof_ratings:
        print(f"Testing with professor rating: {rating}/5.0")
        modified_profile = base_profile.copy()
        modified_profile['prof_rating_override'] = rating
        run_with_modified_profile(modified_profile, 'Professor Rating', rating)
    
    # 7. Professor Difficulty Sensitivity
    print("\n=== PROFESSOR DIFFICULTY SENSITIVITY ===")
    prof_difficulties = [0.5, 1.5, 2.5, 3.5, 4.5]  # Range from easy to very difficult
    
    for difficulty in prof_difficulties:
        print(f"Testing with professor difficulty: {difficulty}/5.0")
        modified_profile = base_profile.copy()
        modified_profile['prof_diff_override'] = difficulty
        run_with_modified_profile(modified_profile, 'Professor Difficulty', difficulty)
    
    # 8. Class Time Sensitivity with early bird preference
    print("\n=== CLASS TIME SENSITIVITY ===")
    # Different class time scenarios
    early_bird = base_profile.get('early_bird', False)
    scenarios_description = "morning" if early_bird else "afternoon/evening"
    
    class_time_scenarios = [
        (["8:30", "9:30", "10:30"], f"All morning classes (best for {scenarios_description} person)"),
        (["12:30", "13:30", "14:30"], "All afternoon classes"),
        (["8:30", "12:30", "16:30"], "Mixed classes")
    ]
    
    for times, description in class_time_scenarios:
        print(f"Testing with {description}")
        modified_profile = base_profile.copy()
        modified_profile['class_time'] = times
        run_with_modified_profile(modified_profile, 'Class Times', description)
    
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
            elif not isinstance(result['value'], (int, float)):
                # Truncate long descriptions
                value_str = value_str[:10] + "..." if len(value_str) > 10 else value_str
                
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


def test_student_profiles():
    """
    Test different student profiles to see how the model responds.
    """
    print("\n=============================================")
    print("TESTING STUDENT PROFILES")
    print("=============================================")
    
    # Prepare simple test profiles
    test_profiles = [
        {
            "name": "Perfect Student",
            "profile": {
                "major": "Life/Health Sciences",
                "overall_grade": 100,
                "course_load": 4,
                "friends_in_class": True,
                "early_bird": True,
                "class_time": ["8:30", "9:30", "10:30"],
                "prerequisite_grade": 100,
                "prereq_grades": {"FAKE101": 100, "FAKE102": 100}
            }
        },
        {
            "name": "Excellent Student",
            "profile": {
                "major": "Life/Health Sciences",
                "overall_grade": 90,
                "course_load": 4,
                "friends_in_class": True,
                "early_bird": True,
                "class_time": ["8:30", "9:30", "10:30"],
                "prerequisite_grade": 92,
                "prereq_grades": {"FAKE101": 90, "FAKE102": 94}
            }
        },
        {
            "name": "Good Student",
            "profile": {
                "major": "Life/Health Sciences",
                "overall_grade": 80,
                "course_load": 5,
                "friends_in_class": True,
                "early_bird": True,
                "class_time": ["8:30", "9:30", "10:30"],
                "prerequisite_grade": 82,
                "prereq_grades": {"FAKE101": 80, "FAKE102": 84}
            }
        },
        {
            "name": "Average Student",
            "profile": {
                "major": "Life/Health Sciences",
                "overall_grade": 70,
                "course_load": 5,
                "friends_in_class": False,
                "early_bird": False,
                "class_time": ["12:30", "14:30", "16:30"],
                "prerequisite_grade": 72,
                "prereq_grades": {"FAKE101": 70, "FAKE102": 74}
            }
        },
        {
            "name": "Struggling Student",
            "profile": {
                "major": "Life/Health Sciences",
                "overall_grade": 60,
                "course_load": 6,
                "friends_in_class": False,
                "early_bird": False,
                "class_time": ["8:30", "9:30", "10:30"],
                "prerequisite_grade": 60,
                "prereq_grades": {"FAKE101": 58, "FAKE102": 62}
            }
        },
        {
            "name": "Failing Student",
            "profile": {
                "major": "Life/Health Sciences",
                "overall_grade": 25,
                "course_load": 7,
                "friends_in_class": False,
                "early_bird": False,
                "class_time": ["8:30", "9:30", "10:30"],
                "prerequisite_grade": 50,
                "prereq_grades": {"FAKE101": 48, "FAKE102": 52}
            }
        }
    ]
    
    # Choose a course code for testing
    course_code = "BIOL369"  # or any valid course code
    
    # Run tests with increased samples for stability
    num_samples = 5000
    
    # Store results
    results = []
    
    # Test each profile
    for test in test_profiles:
        print(f"\nTesting profile: {test['name']}")
        
        # Override professor factors to neutral values to focus on student factors
        test_profile = test["profile"].copy()
        test_profile["prof_rating_override"] = 3.5  # Neutral rating
        test_profile["prof_diff_override"] = 3.0    # Neutral difficulty
        
        # Run prediction
        prediction = run_inference(course_code, test_profile, num_samples)
        
        # Print summary
        print(f"Predicted grade: {prediction['avg_letter']} ({prediction['avg_percent']:.1f}%, GPA: {prediction['avg_gpa']:.2f})")
        
        # Group by letter grade categories
        a_grades = sum(prediction[g] for g in ['A+', 'A', 'A-'])
        b_grades = sum(prediction[g] for g in ['B+', 'B', 'B-'])
        c_grades = sum(prediction[g] for g in ['C+', 'C', 'C-'])
        d_grades = sum(prediction[g] for g in ['D+', 'D', 'D-'])
        f_grade = prediction['F']
        
        print(f"A grades: {a_grades:.2f}, B: {b_grades:.2f}, C: {c_grades:.2f}, D: {d_grades:.2f}, F: {f_grade:.2f}")
        
        # Store results
        results.append({
            "name": test["name"],
            "letter": prediction['avg_letter'],
            "percent": prediction['avg_percent'],
            "gpa": prediction['avg_gpa'],
            "a_prob": a_grades,
            "a_plus_prob": prediction['A+']
        })
    
    # Print comparison table
    print("\n=============================================")
    print("STUDENT PROFILE COMPARISON")
    print("=============================================")
    print(f"{'Student Type':<20} {'Grade':<6} {'Percent':<10} {'GPA':<6} {'A Grades':<10} {'A+ Prob':<10}")
    print("-" * 70)
    
    for r in results:
        print(f"{r['name']:<20} {r['letter']:<6} {r['percent']:<10.1f} {r['gpa']:<6.2f} {r['a_prob']:<10.2f} {r['a_plus_prob']:<10.2f}")
    
    return results

def main():
    print("\n=============================================")
    print("BAYESIAN GRADE PREDICTION SYSTEM")
    print("=============================================")

    course_code, user_profile, prediction = run_grade_prediction()

    # Ask for sensitivity and test students
    response = input("\nRun sensitivity analysis / test student profiles (y/n): ").strip().lower()
    if response.startswith('y'):
        run_sensitivity_analysis(course_code, user_profile)
        test_student_profiles()

    

if __name__ == "__main__":
    main()