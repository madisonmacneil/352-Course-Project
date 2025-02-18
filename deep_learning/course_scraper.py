from selenium import webdriver
from selenium.webdriver.common.by import By 
from webdriver_manager.chrome import ChromeDriverManager
import csv 
import pandas as pd 

#https://www.queensu.ca/academic-calendar/arts-science/course-descriptions/
#https://www.queensu.ca/academic-calendar/business/bachelor-commerce/courses-of-instruction/by20number/

def program_urls():
    driver = webdriver.Chrome()
    driver.get("https://www.queensu.ca/academic-calendar/arts-science/course-descriptions/")

    course_links = []
    links = driver.find_elements(By.CSS_SELECTOR, ".sitemap li a")  
    for li in links:
        course_links.append(li.get_attribute("href"))

    with open("links.csv", 'w') as myfile: 
        wr= csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(course_links)

    driver.quit()

def program_courses(url): 
    driver = webdriver.Chrome()
    driver.get(url)
    course_info_collection = [] 
    courses = driver.find_elements(By.CSS_SELECTOR, ".courseblock")
    for course in courses: 
        course_info = [] 
        #columns = course code, course name, units, description, requirements, offering faculty, learning outcomes 
        course_code = course.find_element(By.CSS_SELECTOR, ".detail-code").text  
        course_name = course.find_element(By.CSS_SELECTOR, ".detail-title").text 
        units = course.find_element(By.CSS_SELECTOR, ".detail-hours_html").text 
        description = course.find_element(By.CSS_SELECTOR, ".courseblockextra").text
        requirements = course.find_element(By.CSS_SELECTOR, ".detail-requirements").text
        faculty = course.find_element(By.CSS_SELECTOR, ".detail-offering_faculty").text
        outcomes = course.find_element(By.CSS_SELECTOR, ".detail-cim_los").text

        course_info.extend([course_code, course_name, units, description, requirements, faculty, outcomes])
        course_info_collection.append(course_info)
    print(course_info_collection)
    driver.quit()

program_courses("https://www.queensu.ca/academic-calendar/arts-science/course-descriptions/anat/")