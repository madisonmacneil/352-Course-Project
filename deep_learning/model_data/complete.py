import pandas as pd
import sqlite3

# Load the CSV
csv_path = 'data/Final_DB.csv'
csv_df = pd.read_csv(csv_path)

# Connect to the existing DB with normalized courses
original_conn = sqlite3.connect('al_courses.db')
normalized_df = pd.read_sql_query("SELECT * FROM new_normalized_courses", original_conn)
original_conn.close()

# Merge CSV and normalized DB data on course_code
merged_df = pd.merge(csv_df, normalized_df, on='course_code', how='inner')

# Save to new database: complete_info.db
new_conn = sqlite3.connect('complete_info.db')
merged_df.to_sql('complete_courses', new_conn, if_exists='replace', index=False)
new_conn.close()

print("✅ Merged data saved to complete_info.db in table 'complete_courses'")
