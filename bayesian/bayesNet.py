import pandas as pd
import numpy as np

# Load course enrollment info (course code, number of students, instructor)
courses_df = pd.read_csv('csp/course_csp_limited.csv')
course_codes = ["APSC111", "APSC171", "APSC182", "CISC101", "HIST105"]
courses_info = courses_df[courses_df['course_code'].isin(course_codes)]
print("\n🗂️ Course Info:")
print(courses_info[['course_code', 'num_students', 'prof']])

# Load instructor quality info
prof_df = pd.read_csv('data/prof_qaulity_info.csv')
prof_df = prof_df.set_index('name')

# Define course → department mapping
dept_map = {
    "APSC111": "Applied Science",
    "APSC171": "Applied Science",
    "APSC182": "Applied Science",
    "CISC101": "Computing & Information Science",
    "HIST105": "History"
}

# Times and days from schedule
earliest_time = {"APSC111": 10.5, "APSC171": 9.5, "APSC182": 8.5,
                 "CISC101": 14.5, "HIST105": 13.5}
first_day = {"APSC111": 1, "APSC171": 1, "APSC182": 2,
             "CISC101": 1, "HIST105": 2}  # 1=Mon, 2=Tue...

# Generate synthetic data for training
data_records = []
np.random.seed(42)

print("\n📊 Generating synthetic student data...\n")
for course in course_codes:
    course_row = courses_info[courses_info['course_code'] == course].iloc[0]
    class_size = course_row['num_students']
    instructor = course_row['prof']
    
    quality = float(prof_df.loc[instructor]['rating_val']) if instructor in prof_df.index else 3.0
    diff = float(prof_df.loc[instructor]['diff_level']) if instructor in prof_df.index else 3.0

    print(f"➡️  Course: {course} | Size: {class_size} | Instructor: {instructor} | Quality: {quality:.1f} | Difficulty: {diff:.1f}")

    for student_id in range(50):
        gpa = np.clip(np.random.normal(3.0, 0.7), 0.0, 4.3)

        # Assign major based on course
        if course.startswith("APSC"):
            major = "Applied Science"
        elif course == "CISC101":
            major = "Computing & Information Science" if np.random.rand() < 0.5 else np.random.choice(list(set(dept_map.values())))
        elif course == "HIST105":
            major = "History" if np.random.rand() < 0.3 else np.random.choice(list(set(dept_map.values())))
        else:
            major = np.random.choice(list(set(dept_map.values())))

        major_related = 1 if dept_map[course] == major else 0
        course_load = np.random.choice([3,4,5,5,5,6])
        class_type = 0

        if class_size > 80:
            participation = np.random.normal(0.4, 0.1)
        elif class_size > 50:
            participation = np.random.normal(0.6, 0.1)
        else:
            participation = np.random.normal(0.8, 0.1)
        participation = float(np.clip(participation, 0.0, 1.0))

        base_apt = gpa / 4.3 + (0.1 if major_related else 0.0)
        aptitude = float(np.clip(np.random.normal(base_apt, 0.1), 0.0, 1.0))

        score = (50 + 1 * earliest_time[course] - 2 * first_day[course] - 5 * class_type
                 - 0.05 * class_size - 2 * course_load + 6 * gpa + 3 * major_related
                 - 4 * diff + 2 * quality + 5 * participation + 12 * aptitude
                 + np.random.normal(0, 5))
        score = np.clip(score, 0, 100)

        if score >= 80:
            grade = 'A'
        elif score >= 70:
            grade = 'B'
        elif score >= 60:
            grade = 'C'
        elif score >= 50:
            grade = 'D'
        else:
            grade = 'F'

        data_records.append({
            "student_id": student_id,
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
            "predicted_score": score,
            "grade": grade
        })

# Convert to DataFrame
data_df = pd.DataFrame(data_records)
print("\n✅ Synthetic data generation complete.")
print("\n📌 Sample data:")
print(data_df.head())
print("\n📈 Grade distribution across all students:")
print(data_df['grade'].value_counts())

# 🔍 Simulate one consistent student across all 5 courses
print("\n🎓 Grade Predictions for a Sample Student's Full Schedule:")

student_major = np.random.choice(list(set(dept_map.values())))
student_gpa = float(np.clip(np.random.normal(3.2, 0.5), 0.0, 4.3))
student_course_load = 5
student_class_type = 0  # in-person
print(f"\n📚 Assigned Major: {student_major} | GPA: {student_gpa:.2f}\n")

for course in course_codes:
    course_row = courses_info[courses_info['course_code'] == course].iloc[0]
    class_size = course_row['num_students']
    instructor = course_row['prof']
    quality = float(prof_df.loc[instructor]['rating_val']) if instructor in prof_df.index else 3.0
    diff = float(prof_df.loc[instructor]['diff_level']) if instructor in prof_df.index else 3.0
    major_related = 1 if dept_map[course] == student_major else 0

    if class_size > 80:
        participation = float(np.clip(np.random.normal(0.4, 0.1), 0.0, 1.0))
    elif class_size > 50:
        participation = float(np.clip(np.random.normal(0.6, 0.1), 0.0, 1.0))
    else:
        participation = float(np.clip(np.random.normal(0.8, 0.1), 0.0, 1.0))

    base_apt = student_gpa / 4.3 + (0.1 if major_related else 0.0)
    aptitude = float(np.clip(np.random.normal(base_apt, 0.1), 0.0, 1.0))

    score = (50 + 1 * earliest_time[course] - 2 * first_day[course] - 5 * student_class_type
             - 0.05 * class_size - 2 * student_course_load + 6 * student_gpa + 3 * major_related
             - 4 * diff + 2 * quality + 5 * participation + 12 * aptitude
             + np.random.normal(0, 5))
    score = np.clip(score, 0, 100)

    if score >= 80:
        grade = 'A'
    elif score >= 70:
        grade = 'B'
    elif score >= 60:
        grade = 'C'
    elif score >= 50:
        grade = 'D'
    else:
        grade = 'F'

    print(f"\n📘 {course}")
    print(f"{'Class Size:':>20} {class_size}")
    print(f"{'Instructor:':>20} {instructor}")
    print(f"{'Difficulty:':>20} {diff:.2f}")
    print(f"{'Quality:':>20} {quality:.2f}")
    print(f"{'Major Related:':>20} {'Yes' if major_related else 'No'}")
    print(f"{'Participation:':>20} {participation:.2f}")
    print(f"{'Aptitude:':>20} {aptitude:.2f}")
    print(f"{'Predicted Score:':>20} {score:.1f}")
    print(f"{'Predicted Grade:':>20} {grade}")
