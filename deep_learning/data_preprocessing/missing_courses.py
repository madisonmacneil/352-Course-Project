'''
This file finds any courses not included in the DB
'''
import pandas as pd
import sqlite3

# Load the CSV file
csv_df = pd.read_csv('data/courses_instructors.csv')

# Connect to the SQLite DB
conn = sqlite3.connect('data/sql_dbs/limited_attributes.db')

# Load course codes from new_normalized_courses table
sql_df = pd.read_sql_query("SELECT course_code FROM new_normalized_courses", conn)

# Close the connection
conn.close()

# Find course codes in CSV but not in DB
missing_course_codes = csv_df[~csv_df['course_code'].isin(sql_df['course_code'])]

# Save the missing codes to a new CSV file
missing_course_codes.to_csv('data/sql_dbs/missing_course_codes.csv', index=False)

print("✅ Saved missing course codes to 'data/sql_dbs/missing_course_codes.csv'")
