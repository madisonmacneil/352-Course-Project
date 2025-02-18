from selenium import webdriver
from selenium.webdriver.common.by import By 
from webdriver_manager.chrome import ChromeDriverManager
import numpy as np 
#https://www.queensu.ca/academic-calendar/arts-science/course-descriptions/
#https://www.queensu.ca/academic-calendar/business/bachelor-commerce/courses-of-instruction/by20number/

driver = webdriver.Chrome()
driver.get("https://www.queensu.ca/academic-calendar/arts-science/course-descriptions/")

course_links = []
links = driver.find_elements(By.CSS_SELECTOR, ".sitemap li a")  # Finds li inside sitemap
for li in links:
    print(li.get_attribute("href")) 
    course_links.append(li.get_attribute("href"))
driver.quit()
