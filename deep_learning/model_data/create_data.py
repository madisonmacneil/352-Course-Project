import pandas as pd 

'''
This file creates all the training/test data for the seq2seq model to train on by creating a series of potential nl queries that the model may 
receive as input and their sql equivalents 
'''
df = pd.read_csv('')
#read unique department values 


# need synonyms 

#{sentence prefix: show me, I want to see, list all, what are the,which, find all, i need, get me, what are the etc.... nl permutations}
#courses taught by instructor X 

#{sentence prefix} courses without any prerequisites

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

#Which {year} {department} courses are full year? 

#{prefix} {unit_num}-unit {department/ faculty} courses