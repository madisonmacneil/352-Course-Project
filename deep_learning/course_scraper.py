from selenium import webdriver
from selenium.webdriver.common.by import By 
import csv 
import pandas as pd 
import os 

def program_urls(path):
    driver = webdriver.Chrome()
    driver.get(path)

    course_links = []
    links = driver.find_elements(By.CSS_SELECTOR, ".sitemap li a")  
    for li in links:
        course_links.append(li.get_attribute("href"))

    with open("links.csv", 'a') as myfile: 
        wr= csv.writer(myfile, quoting=csv.QUOTE_ALL)
        for link in course_links:
            wr.writerow([link])

    driver.quit()

def program_courses(url): 
    driver = webdriver.Chrome()
    driver.get(url)
    course_info_collection = [] 
    courses = driver.find_elements(By.CSS_SELECTOR, ".courseblock")
    requirements = None
    for course in courses: 
        course_info = [] 

        #columns = course code, course name, units, description, requirements, offering faculty, learning outcomes 
        course_code = course.find_element(By.CSS_SELECTOR, ".detail-code").text  
        course_name = course.find_element(By.CSS_SELECTOR, ".detail-title").text 
        units = course.find_element(By.CSS_SELECTOR, ".detail-hours_html").text 
        description = course.find_element(By.CSS_SELECTOR, ".courseblockextra").text
        try: requirements = course.find_element(By.CSS_SELECTOR, ".detail-requirements").text
        except: 
            pass
        faculty = course.find_element(By.CSS_SELECTOR, ".detail-offering_faculty").text
        outcomes = course.find_element(By.CSS_SELECTOR, ".detail-cim_los").text
        
        course_info.extend([course_code, course_name, units, description, requirements, faculty, outcomes])
        course_info_collection.append(course_info)

    df = pd.DataFrame(course_info_collection, columns = ['course_code', 'course_name', 'units', 'description', 'requirements', 'faculty', 'outcomes'])
    output_path = 'db.csv'
    df.to_csv(output_path, mode ='a', header = not os.path.exists(output_path))
    driver.quit()


def courses_url_by_faculty(faculty_urls):
    for i in range(len(faculty_urls)): 
        program_urls(faculty_urls[i])

def course_info_scrape(my_file): 
    with open(my_file, 'r') as f:
        reader = csv.reader(f)
        for row in reader: 
            print(row[0])
            program_courses(row[0])

#courses_url_by_faculty(['https://www.queensu.ca/academic-calendar/arts-science/course-descriptions/', 'https://www.queensu.ca/academic-calendar/engineering-applied-sciences/courses-instruction/'])

course_info_scrape('courses.csv')

