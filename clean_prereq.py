import pandas as pd
import re

# Load the CSV
df = pd.read_csv("bayesian/complete_courses.csv")

def clean_prereq_list(prereq_str):
    if not isinstance(prereq_str, str) or prereq_str.strip() == "":
        return ""
    
    # Split comma-separated prereqs
    prereqs = [p.strip() for p in prereq_str.split(',') if p.strip()]
    cleaned = []

    for course in prereqs:
        # Remove anything after a slash
        course = course.split('/')[0].strip()

        # Fix formatting: ensure space between dept and number
        match = re.match(r'^([A-Z]{3,5})\s*?(\d{3})$', course.replace(" ", ""))
        if match:
            dept, num = match.groups()
            cleaned.append(f"{dept} {num}")

    return ', '.join(cleaned)

# Apply cleaning function
df['prereq_codes'] = df['prereq_codes'].apply(clean_prereq_list)

# Save cleaned CSV
df.to_csv("bayesian/complete_courses_cleaned.csv", index=False)
