import sqlite3
import pandas as pd
import json
import random
import re
import itertools

def generate_training_data(db_path='data/sql_dbs/limited_attributes.db'):
    """Generate a training dataset of natural language queries and corresponding SQL queries."""
    
    # Connect to the database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    raw_instructors = cursor.execute("""
        SELECT instructor 
        FROM complete_courses 
        WHERE instructor IS NOT NULL AND instructor != ''
    """).fetchall()

    # Parse JSON strings and flatten
    unique_instructors = set()
    for row in raw_instructors:
        try:
            instructors_list = json.loads(row[0])
            unique_instructors.update(instructors_list)
        except json.JSONDecodeError:
            print(f"Skipping invalid JSON: {row[0]}")

    # Optional: sort alphabetically
    instructors = sorted(unique_instructors)

    # Fetch unique values from the database to use in templates

    dept_codes = [row[0] for row in cursor.execute("SELECT DISTINCT department_code FROM complete_courses WHERE department_code IS NOT NULL AND department_code != ''").fetchall()]
    dept_names = [row[0] for row in cursor.execute("SELECT DISTINCT department_name FROM complete_courses WHERE department_name IS NOT NULL AND department_name != ''").fetchall()]
    course_codes = [row[0] for row in cursor.execute("SELECT DISTINCT course_code FROM complete_courses WHERE course_code IS NOT NULL AND course_code != ''").fetchall()]
    faculties = [row[0] for row in cursor.execute("SELECT DISTINCT faculty FROM complete_courses WHERE faculty IS NOT NULL AND faculty != ''").fetchall()]
    
    print(f"instructor length {len(instructors)}")
    # Try to extract keywords from description or dedicated keywords column
    try:
        keywords_raw = cursor.execute("SELECT keywords FROM complete_courses WHERE keywords IS NOT NULL AND keywords != '' LIMIT 100").fetchall()
        keywords = []
        for k in keywords_raw:
            # Handle both string and JSON formatted keywords
            try:
                if k[0].startswith('['):
                    parsed = json.loads(k[0])
                    keywords.extend(parsed)
                else:
                    keywords.extend([kw.strip() for kw in k[0].split(',')])
            except:
                if k[0]:
                    keywords.append(k[0])
        keywords = list(set(keywords))[:300]  # Limit to 50 unique keywords

    except:
        # If keywords column doesn't exist, extract from descriptions
        descriptions = cursor.execute("SELECT description FROM complete_courses WHERE description IS NOT NULL AND description != '' LIMIT 50").fetchall()
        common_words = ["and", "the", "of", "to", "a", "in", "for", "on", "with", "by", "at", "from", "as", "an", "is", "are"]
        keywords = []
        for desc in descriptions:
            if desc[0]:
                words = re.findall(r'\b\w+\b', desc[0].lower())
                keywords.extend([w for w in words if len(w) > 5 and w not in common_words])
        keywords = list(set(keywords))[:50]  # Limit to 50 unique keywords
    
    # Prefixes for more natural language variation
    prefixes = ["", "show me", "I want to see", "list all", "what are the", "which", "find all", "I need", "get me", "what are the"]
    
    # Years for course levels
    years = ['first', 'second', 'third', 'fourth', '1st', '2nd', '3rd', '4th']
    
    # Template pairs mapping natural language queries to SQL queries
    templates = [
        # Basic templates
        {
            "nl_template": "{prefix} courses taught by {instructor}",
            "sql_template": "SELECT course_code course_name description FROM complete_courses WHERE instructor = '{instructor}'"
        },
        {
            "nl_template": "{prefix} {year} courses taught by {instructor}",
            "sql_template": "SELECT course_code course_name description FROM complete_courses WHERE instructor = '{instructor}' AND year = '{year_normalized}'"
        },
        {
            "nl_template": "{prefix} courses without any prerequisites",
            "sql_template": "SELECT course_code course_name description FROM complete_courses WHERE prereq_codes IS NULL"
        },
        {
            "nl_template": "{prefix} {year} year courses without any prerequisites",
            "sql_template": "SELECT course_code course_name description FROM complete_courses WHERE year = '{year_normalized}' AND prereq_codes IS NULL"
        },
        {
            "nl_template": "{prefix} {year} year {dept_code} courses without any prerequisites",
            "sql_template": "SELECT course_code course_name description FROM complete_courses WHERE year = '{year_normalized}' AND department_code = '{dept_code}' AND prereq_codes IS NULL"
        },
        {
            "nl_template": "{prefix} {year} year {dept_name} courses without any prerequisites",
            "sql_template": "SELECT course_code course_name description FROM complete_courses WHERE year = '{year_normalized}' AND department_name = '{dept_name}' AND prereq_codes IS NULL"
        },
                {
            "nl_template": "{prefix} {year} year {dept_name} courses",
            "sql_template": "SELECT course_code course_name description FROM complete_courses WHERE year = '{year_normalized}' AND department_name = '{dept_name}'"
        },
        {
            "nl_template": "{prefix} exclusions of {course_code}",
            "sql_template": "SELECT exclusions FROM complete_courses WHERE course_code = '{course_code}'"
        },
        {
            "nl_template": "{prefix} courses with {course_code} as a prerequisite",
            "sql_template": "SELECT course_code course_name description FROM complete_courses WHERE prereq_codes LIKE '{course_code}'"
        },
        {
            "nl_template": "{prefix} courses I can't take if I take {course_code}",
            "sql_template": "SELECT course_code course_name description FROM complete_courses WHERE exclusions LIKE '{course_code}'"
        },
        {
            "nl_template": "{prefix} courses I need to take {course_code}",
            "sql_template": "SELECT prereq_codes FROM complete_courses WHERE course_code = '{course_code}'"
        },
        {
            "nl_template": "{prefix} {faculty} courses",
            "sql_template": "SELECT course_code course_name description FROM complete_courses WHERE faculty = '{faculty}'"
        },
        {
            "nl_template": "{prefix} {year} year {faculty} courses",
            "sql_template": "SELECT course_code course_name description FROM complete_courses WHERE faculty = '{faculty}' AND year = {year_normalized}"
        },
        {
            "nl_template": "{prefix} {year} year {dept_name} courses",
            "sql_template": "SELECT course_code course_name description FROM complete_courses WHERE  dept_name = {dept_name} AND year = '{year_normalized}'"
        },
        {
            "nl_template": "{prefix} {dept_code} courses",
            "sql_template": "SELECT course_code course_name description FROM complete_courses WHERE department_code = '{dept_code}'"
        },
        {
            "nl_template": "{prefix} full year courses",
            "sql_template": "SELECT course_code course_name description FROM complete_courses WHERE units = 6.0"
        },
        {
            "nl_template": "{prefix} full year {dept_code} courses",
            "sql_template": "SELECT course_code course_name description FROM complete_courses WHERE department_code = '{dept_code}' AND units = 6.0"
        },
        {
            "nl_template": "{prefix} full year {faculty} courses",
            "sql_template": "SELECT course_code course_name description FROM complete_courses WHERE faculty = '{faculty}' AND units = 6.0;"
        },
        {
            "nl_template": "{prefix} prerequisites of {course_code}",
            "sql_template": "SELECT prereq_codes FROM complete_courses WHERE course_code = '{course_code}'"
        },
        {
            "nl_template": "{prefix} {year} year courses",
            "sql_template": "SELECT course_code course_name description FROM complete_courses WHERE year = '{year_normalized}'"
        }
    ]
    
    # Generate the training data
    training_data = []
    
    # Function to normalize year format
    def normalize_year(year_str):
        mapping = {
        '1st': 1, 'first': 1,
        '2nd': 2, 'second': 2,
        '3rd': 3, 'third': 3,
        '4th': 4, 'fourth': 4
    }
        return mapping.get(year_str, year_str)
    
    # For reproducibility
    random.seed(42)
    
    # Generate a limited but diverse set of examples
    for template in templates:
        # Decide how many examples to generate for this template
        num_examples = 400
        
        for _ in range(num_examples):
            prefix = random.choice(prefixes)
            
            # Create variations based on the template requirements
            if "{instructor}" in template["nl_template"]:
                if instructors:
                    instructor = random.choice(instructors)
                    nl_query = template["nl_template"].format(prefix=prefix, instructor=instructor)
                    sql_query = template["sql_template"].format(instructor=instructor)
                    training_data.append({"natural_language_query": nl_query.strip(), "sql_query": sql_query})
            
            elif "{year}" in template["nl_template"] and "{course_code}" in template["nl_template"]:
                if years and course_codes:
                    year = random.choice(years)
                    course_code = random.choice(course_codes)
                    nl_query = template["nl_template"].format(prefix=prefix, year=year, course_code=course_code)
                    sql_query = template["sql_template"].format(year_normalized=normalize_year(year), course_code=course_code)
                    training_data.append({"natural_language_query": nl_query.strip(), "sql_query": sql_query})
            
            elif "{year}" in template["nl_template"] and "{dept_code}" in template["nl_template"]:
                if years and dept_codes:
                    year = random.choice(years)
                    dept_code = random.choice(dept_codes)
                    nl_query = template["nl_template"].format(prefix=prefix, year=year, dept_code=dept_code)
                    sql_query = template["sql_template"].format(year_normalized=normalize_year(year), dept_code=dept_code)
                    training_data.append({"natural_language_query": nl_query.strip(), "sql_query": sql_query})
            
            elif "{year}" in template["nl_template"] and "{dept_name}" in template["nl_template"]:
                if years and dept_names:
                    year = random.choice(years)
                    dept_name = random.choice(dept_names)
                    nl_query = template["nl_template"].format(prefix=prefix, year=year, dept_name=dept_name)
                    sql_query = template["sql_template"].format(year_normalized=normalize_year(year), dept_name=dept_name)
                    training_data.append({"natural_language_query": nl_query.strip(), "sql_query": sql_query})
            
            elif "{course_code}" in template["nl_template"] and "exclusions" in template["nl_template"]:
                if course_codes:
                    course_code = random.choice(course_codes)
                    nl_query = template["nl_template"].format(prefix=prefix, course_code=course_code)
                    sql_query = template["sql_template"].format(course_code=course_code)
                    training_data.append({"natural_language_query": nl_query.strip(), "sql_query": sql_query})
            
            elif "{course_code}" in template["nl_template"]:
                if course_codes:
                    course_code = random.choice(course_codes)
                    nl_query = template["nl_template"].format(prefix=prefix, course_code=course_code)
                    sql_query = template["sql_template"].format(course_code=course_code)
                    training_data.append({"natural_language_query": nl_query.strip(), "sql_query": sql_query})
            elif "{year}" in template["nl_template"] and "{faculty}" in template["nl_template"]:
                if faculties:
                    year = random.choice(years)
                    faculty = random.choice(faculties)
                    nl_query = template["nl_template"].format(prefix=prefix, faculty=faculty, year=year)
                    sql_query = template["sql_template"].format(faculty=faculty, year_normalized=normalize_year(year))
                    training_data.append({"natural_language_query": nl_query.strip(), "sql_query": sql_query})
            
            elif "{faculty}" in template["nl_template"]:
                if faculties:
                    faculty = random.choice(faculties)
                    nl_query = template["nl_template"].format(prefix=prefix, faculty=faculty)
                    sql_query = template["sql_template"].format(faculty=faculty)
                    training_data.append({"natural_language_query": nl_query.strip(), "sql_query": sql_query})

            elif "{dept_code}" in template["nl_template"]:
                if dept_codes:
                    dept_code = random.choice(dept_codes)
                    nl_query = template["nl_template"].format(prefix=prefix, dept_code=dept_code)
                    sql_query = template["sql_template"].format(dept_code=dept_code)
                    training_data.append({"natural_language_query": nl_query.strip(), "sql_query": sql_query})
            
            elif "{year}" in template["nl_template"]:
                if years:
                    year = random.choice(years)
                    nl_query = template["nl_template"].format(prefix=prefix, year=year)
                    sql_query = template["sql_template"].format(year_normalized=normalize_year(year))
                    training_data.append({"natural_language_query": nl_query.strip(), "sql_query": sql_query})
            
            else:  # Templates without replaceable parameters
                nl_query = template["nl_template"].format(prefix=prefix)
                sql_query = template["sql_template"]
                training_data.append({"natural_language_query": nl_query.strip(), "sql_query": sql_query})
    
    # Close the database connection
    conn.close()
    
    # Convert to DataFrame
    df = pd.DataFrame(training_data)
    
    # Remove duplicates
    df = df.drop_duplicates()
    
    return df

def save_training_data(df, output_path='data/course_query_training_data.csv'):
    """Save the training data to a CSV file."""
    df.to_csv(output_path, index=False)
    print(f"Training data saved to {output_path}")
    print(f"Generated {len(df)} unique query pairs")
    
    # Show a few examples
    print("\nSample of generated data:")
    sample = df.sample(min(5, len(df)))
    for i, row in sample.iterrows():
        print(f"\nNatural Language Query: {row['natural_language_query']}")
        print(f"SQL Query: {row['sql_query']}")

# Generate and save the training data
if __name__ == "__main__":
    df = generate_training_data()
    save_training_data(df)