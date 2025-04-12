import sqlite3
import pandas as pd

# Load the CSV (if needed)
csv_path = 'data/Final_DB.csv'
df_csv = pd.read_csv(csv_path)

# Connect to the existing DB
original_conn = sqlite3.connect('complete_info.db')

# Query with properly quoted column name
query = """
SELECT course_code, course_name, units, year, description, requirements, 
       faculty, outcomes, corequisites, "exclusions.complete_exclusions" AS exclusions, 
       recommended, department_name, department_code, instructor, keywords, prereq_codes
FROM complete_courses
"""

# Execute and fetch into DataFrame
df = pd.read_sql_query(query, original_conn)
original_conn.close()

# Write to new DB
new_conn = sqlite3.connect('new_courses.db')
df.to_sql('complete_courses', new_conn, if_exists='replace', index=False)
new_conn.close()

print("✅ Data written to new_courses.db in table 'complete_courses'.")
