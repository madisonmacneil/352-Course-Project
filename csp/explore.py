"""
Some faculties have more than 15 courses of of the same year. For example there are 25 HIST 200 courses.
Given our constraint of no courses from the same faculty and same year running at the same time there is a max of 15 slots for courses of the same year and faculty. 
To accomodate for this any courses that make this constraint impossible, We remove random courses so there is a max of 10 courses for each year and faculty. 
Thus there will be 10 random HIST 200 classes in the dataset. I chose 10 instead of 15 to give some breathing room and to allow the CSP to have more solutions. 
"""

import pandas as pd
import re
import random

random.seed(1)

# Function to extract department code
def extract_department(course_code):
    match = re.search(r'([A-Z]+)', course_code)
    if match:
        return match.group(1)
    return "Unknown"

# Function to extract year level
def extract_year(course_code):
    match = re.search(r'([A-Z]+)(\d)', course_code)
    if match:
        return int(match.group(2))
    return 0

# Load the courses CSV
courses_df = pd.read_csv('352-Course-Project/csp/course_csp.csv')

# Add department and year columns for easier filtering
courses_df['department'] = courses_df['course_code'].apply(extract_department)
courses_df['year'] = courses_df['course_code'].apply(extract_year)

# Group by department and year to identify those with more than 10 courses
dept_year_counts = courses_df.groupby(['department', 'year']).size()
excess_groups = [(dept, year) for (dept, year), count in dept_year_counts.items() if count > 10]

# Track statistics for reporting
removed_courses = []
total_removed = 0
affected_dept_years = []

# Process each department+year combination that exceeds 10 courses
for dept, year in excess_groups:
    # Get all courses for this department+year
    dept_year_df = courses_df[(courses_df['department'] == dept) & (courses_df['year'] == year)]
    
    # Calculate how many to remove
    current_count = len(dept_year_df)
    to_remove_count = current_count - 10
    
    # Randomly select indices to remove
    indices_to_remove = random.sample(list(dept_year_df.index), to_remove_count)
    
    # Store information about removed courses for reporting
    removed_from_group = courses_df.loc[indices_to_remove, ['course_code', 'name', 'prof']].values.tolist()
    removed_courses.extend(removed_from_group)
    total_removed += to_remove_count
    
    # Update statistics
    affected_dept_years.append(f"{dept}{year} (removed {to_remove_count} of {current_count})")
    
    # Remove the selected courses
    courses_df = courses_df.drop(indices_to_remove)

# Save the modified dataframe
courses_df.drop(['department', 'year'], axis=1, inplace=True)  # Remove helper columns
courses_df.to_csv('352-Course-Project/csp/course_csp_limited.csv', index=False)

# # Print summary
# print(f"\nModified dataset saved to: course_csp_limited.csv")
# print(f"Total courses removed: {total_removed}")
# print(f"Affected department-year combinations: {len(affected_dept_years)}")
# for dept_year in affected_dept_years:
#     print(f"  - {dept_year}")

# # print details of removed courses
# print("\nRemoved courses (first 10 shown):")
# for i, (code, name, prof) in enumerate(removed_courses[:10]):
#     print(f"  {code}: {name} (Prof: {prof})")
# if len(removed_courses) > 10:
#     print(f"  ...and {len(removed_courses) - 10} more")

# Verify the changes: no department-year should have more than 10 courses now
courses_df['department'] = courses_df['course_code'].apply(extract_department)
courses_df['year'] = courses_df['course_code'].apply(extract_year)
new_counts = courses_df.groupby(['department', 'year']).size()
max_count = new_counts.max()
print(f"\nVerification: Maximum courses per department-year in new dataset: {max_count}")
print("All department-year combinations now have 10 or fewer courses.")