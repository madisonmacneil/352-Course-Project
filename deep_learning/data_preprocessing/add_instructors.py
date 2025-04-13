'''
This file processes the scraped professor data, adding only the courses that exist in the DB and the professors 
who teach a course that exists in the DB. Appending their name to the associated course in a new DB attribute 'instructors'. 
'''
import pandas as pd 
import ast
import re 

def only_valid_courses():
    courses_df = pd.read_csv('data/process_csvs/instructor_info/v2_requirements_parsed.csv')

    print(courses_df.head(5))

    course_codes = courses_df['course_code'].tolist()
    for i in range(len(course_codes)):
        course_codes[i] = course_codes[i].replace(' ','')

    rmp_courses = pd.read_csv('data/process_csvs/instructor_info/v1_teaches_courses_raw.csv')

    rmp_classes = rmp_courses['courses_taught'].to_list()

    for i in range(len(rmp_classes)): 
        courses = ast.literal_eval(rmp_classes[i])
        print(courses)
        courses_to_remove = []
        course_codes_lower = {code.lower() for code in course_codes}
        for j in range(len(courses)): 
            if courses[j].lower() not in course_codes_lower:
                print('removing' , courses[j])
                courses_to_remove.append(courses[j])
        for val in courses_to_remove:
            print(val)
            courses.remove(val)
        rmp_classes[i] = courses
        print(rmp_classes[i])

    rmp_courses['courses_taught'] = rmp_classes
    rmp_courses.to_csv('data/process_csvs/instructor_info/v2_teaches_course_valid.csv', index= False)

def remove_profs_without_courses(): 
    '''
    This function removes any professor from the prof DB that does not have any associated courses.
    '''
    df = pd.read_csv('data/process_csvs/instructor_info/v2_teaches_course_valid.csv')
     # Convert 'courses_taught' column from string to list (if necessary)
    df['courses_taught'] = df['courses_taught'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    # Remove rows where 'courses_taught' is an empty list
    df = df[df['courses_taught'].apply(lambda x: len(x) > 0)]

    print(df.head(10))  # Check output before saving
    df.to_csv('data/process_csvs/instructor_info/v2_teaches_course_valid.csv', index=False)


def course_cross_check(): 
    '''
    Confirm that every course that appears in the DB of profs from Rate my Prof, also appears in our DB of courses from the Queen's Official Website. 
    '''

    df = pd.read_csv('data/process_csvs/instructor_info/v2_teaches_course_valid.csv')
    rmp_classes = df['courses_taught'].to_list()
    all_rmp = []
    for i in range(len(rmp_classes)): 
        courses = ast.literal_eval(rmp_classes[i])
        for j in range(len(courses)): 
            all_rmp.append(courses[j])

    courses_df = pd.read_csv('v2_db_requirements_parsed.csv')
    course_codes = courses_df['course_code'].tolist()

    for i in range(len(course_codes)):
        course_codes[i] = course_codes[i].replace(' ','')
    course_codes_lower = {code.lower() for code in course_codes}
    all_rmp_lower = {code.lower() for code in all_rmp}
    courses_wo_profs =[]
    courses_w_profs = []
    for course in course_codes_lower: 
        if course not in all_rmp_lower: 
            courses_wo_profs.append(course)
            print(course)
        else:
            courses_w_profs.append(course)

    print(len(courses_w_profs))
    print(len(all_rmp))
    print(len(courses_wo_profs))

def instructors_to_main_db():
    main_df = pd.read_csv('data/process_csvs/course_info/v3_db_new_attributes.csv')
    prof_df = pd.read_csv('data/process_csvs/instructor_info/v2_teaches_course_valid.csv')

    courses_taught_by = {}
    count = 0 
    for index, row in prof_df.iterrows():
        name = row['name'] 
        courses = row['courses_taught'][1:-1].replace("'","").upper().split(',')
        courses = [course.lstrip() for course in courses]

        chat_courses = [course.strip() for course in row['courses_taught'][1:-1].replace("'", "").upper().split(',')]

        for course in courses: 
            count += 1 

        for course in courses: 
            match = re.search(r"\d", course)
            if match:
                index = match.start()
                course = course[:index] + ' ' + course[index:]  # Ensure space between letters and numbers
                if course in courses_taught_by: 
                    courses_taught_by[course].append(name)
                else: 
                    courses_taught_by[course] = [name]
            else:
                print(f"Skipping invalid course format: {course}")

    main_df['instructor'] = None  # Create an empty instructor column if it doesn't exist

    for index, row in main_df.iterrows():
        course_code = row['course_code'].strip().upper()
        if course_code in courses_taught_by:
            main_df.at[index, 'instructor'] = ', '.join(courses_taught_by[course_code])
    
    # print(courses_taught_by)
    print(f"Courses Processed: {count}")


    main_df.to_csv('data/courses_instructors.csv', index = False)

instructors_to_main_db()


    


