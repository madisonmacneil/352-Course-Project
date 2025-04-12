#read unique department values 
prefixes = ["", "show me", "I want to see", "list all", "what are the" , "which", "find all", "I need", "get me","what are the"]
year = ['first', 'second', 'third','fourth', '1st', '2nd','3rd', '4th']
sentences =['courses taught by {instructor}','courses without any prerequisites', '{year} year courses without any prereqs',
            '{year} year courses without any prerequisites', '{year} year {course_code} courses without any prereqs', 
            '{year} year {course_code} courses without any prerequisites', '{year} year {dept_code} courses without any prereqs', 
            '{year} year {dept_code} courses without any prerequisites', '{year} year {dept_name} courses without any prereqs', 
            '{year} year {dept_name} courses without any prerequisites', 'exclusions of {course_code}', 
            'courses with {course_code} as a prerequisites', 'courses about {keyword}', "courses I can't take if I take {course_code}", 
            'courses I need to take {course_code}', "{faculty} courses", "{dept_code} courses", "full year courses", "full year {dept_code} courses", 
            "full year {faculty} courses", "prerequisites of {course_code}", "{year} year courses"
              ]

sql = ["SELECT * FROM new_normalized_courses WHERE instructor = {instructor};", "SELECT * FROM new_normalized_courses WHERE prereq_codes = '[]';",
        "SELECT * FROM new_normalized_courses WHERE year = 'first' AND prereq_codes = '[]'" ]
#{sentence prefix: show me, I want to see, list all, what are the,which, find all, i need, get me, what are the etc.... nl permutations}
#{sentence prefix} courses taught by {instructor}

#{sentence prefix} courses without any prerequisites
#courses 
#{sentence prefix} {first, second, third, fourth} year courses without any prereqs

#{sentence prefix} {department code or department name or faculty } courses without any prereqs 

#{sentence prefix} {first, second, third, fourth} year {course code or department or faculty } courses without any prereqs 
#{sentence prefix}  {course code or department or faculty } courses in {first, second, third, fourth} year 


#{sentence prefix} {first, second, third, fourth} year {course code or department or faculty } courses 

#{sentence prefix} all the courses offered by the {department name } faculty 


#What courses require {course_code} as a prerequisite? 

#What are the exclusions of {course_code}? 

#Which classes can I not take if I take {course_code}

#Which {year} courses are full year?  
    #SQL: units = 6.0 

#Which {year} {department} courses are full year? 
        #SQL: units = 6.0 

#{prefix} {unit_num}-unit {department/ faculty} courses
        #SQL: units = 6.0 