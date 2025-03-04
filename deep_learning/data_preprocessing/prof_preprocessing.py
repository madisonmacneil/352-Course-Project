import pandas as pd 
import ast

def only_valid_courses():
    courses_df = pd.read_csv('courses_db.csv')

    print(courses_df.head(5))

    course_codes = courses_df['course_code'].tolist()
    for i in range(len(course_codes)):
        course_codes[i] = course_codes[i].replace(' ','')

    rmp_courses = pd.read_csv('prof_classes.csv')

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
    rmp_courses.to_csv('prof_teaches.csv', index= False)

def remove_profs_without_courses(): 
    '''
    This function removes any professor from the prof DB that does not have any associated courses.
    '''
    df = pd.read_csv('prof_teaches.csv')
     # Convert 'courses_taught' column from string to list (if necessary)
    df['courses_taught'] = df['courses_taught'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    # Remove rows where 'courses_taught' is an empty list
    df = df[df['courses_taught'].apply(lambda x: len(x) > 0)]

    print(df.head(10))  # Check output before saving
    df.to_csv('prof_teaches.csv', index=False)

def course_cross_check(): 
    '''
    Confirm that every course that appears in the DB of profs from Rate my Prof, also appears in our DB of courses from the Queen's Official Website. 
    '''

    df = pd.read_csv('prof_teaches.csv')
    rmp_classes = df['courses_taught'].to_list()
    all_rmp = []
    for i in range(len(rmp_classes)): 
        courses = ast.literal_eval(rmp_classes[i])
        for j in range(len(courses)): 
            all_rmp.append(courses[j])

    courses_df = pd.read_csv('courses_db.csv')
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


