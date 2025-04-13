
from selenium import webdriver
from selenium.webdriver.common.by import By 
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import StaleElementReferenceException
from selenium.common.exceptions import StaleElementReferenceException, TimeoutException, NoSuchElementException
import time
import pandas as pd 
import os

with open('data/process_csvs/intructor_info/teacher_links.txt', 'r') as f: 
    prof_pages = f.readlines()

driver = webdriver.Safari()
all_prof_info = []

num = 1
try:
    for page in prof_pages[210:]:
        count = 0 
        print(num)
        num += 1 
        prof_info = []
        driver.get(page)

        # Close modal popup if present
        try:
            close_button = WebDriverWait(driver, 3).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, "div.FullPageModal__StyledFullPageModal-sc-1tziext-1 button.Buttons__Button-sc-19xdot-1.CCPAModal__StyledCloseButton-sc-10x9kq-2"))
        )
            close_button.click()

        except (TimeoutException, NoSuchElementException):
            pass  # Ignore if the modal doesn't appear

        # Get professor name
        try:
            prof_name = driver.find_element(By.XPATH, "//h1[contains(@class, 'NameTitle__NameWrapper-dowf0z-2') and contains(@class, 'fEoACI')]").text
            prof_info.append(prof_name)
        except NoSuchElementException:
            print(f"Professor name not found on page: {page}")
            continue  # Skip this page if no name is found

        # Extract existing reviews first
        reviews = driver.find_elements(By.XPATH, "//div[contains(@class, 'RatingHeader__StyledHeader-sc-1dlkqw1-1')]")
        found_old_review = False
        prof_courses = set()


        for review in reviews:
            try:
                date = review.find_element(By.CLASS_NAME, "TimeStamp__StyledTimeStamp-sc-9q2r30-0").text
                year = int(date.split(', ')[1])  # Extract year

                if year < 2023:
                    found_old_review = True
                    break  # Stop processing if review is from 2020 or earlier
            
                course_code = review.find_element(By.XPATH, ".//div[contains(@class, 'RatingHeader__StyledClass-sc-1dlkqw1-3')]").text.strip()
                prof_courses.add(course_code)

            except (NoSuchElementException, IndexError, ValueError):
                continue  # Skip reviews that don't contain date or course code

        # Only click "Load More Ratings" if no old reviews were found
        while not found_old_review:
            count += 20
            try:
                load_more_btn = driver.find_element(By.XPATH, "//button[text()='Load More Ratings']")
                driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", load_more_btn)
                load_more_btn.click()
                time.sleep(1)  # Allow time for new content to load
            
                # Re-check reviews after loading more
                reviews = driver.find_elements(By.XPATH, "//div[contains(@class, 'RatingHeader__StyledHeader-sc-1dlkqw1-1')]")
                for review in reviews[count:]:
                    try:
                        date = review.find_element(By.CLASS_NAME, "TimeStamp__StyledTimeStamp-sc-9q2r30-0").text
                        year = int(date.split(', ')[1])

                        if year < 2023:
                            found_old_review = True
                            break  # Stop clicking if old review is found

                        course_code = review.find_element(By.XPATH, ".//div[contains(@class, 'RatingHeader__StyledClass-sc-1dlkqw1-3')]").text.strip()
                        prof_courses.add(course_code)

                    except (NoSuchElementException, IndexError, ValueError):
                        continue  # Skip problematic reviews
            except (NoSuchElementException, StaleElementReferenceException):
                break  # Stop clicking if button disappears

        prof_info.append(list(prof_courses))  # Convert set to list before storing
        all_prof_info.append(prof_info)
        print(prof_info)
        
except KeyboardInterrupt:
    print("\nProcess interrupted. Exiting loop.")

# Save extracted data to CSV
df = pd.DataFrame(all_prof_info, columns=['name', 'courses_taught'])
output_path = 'data/process_csvs/instructor_info/v1_teaches_course_raw.csv'
df.to_csv(output_path, mode ='a', header = not os.path.exists(output_path))



driver.quit()