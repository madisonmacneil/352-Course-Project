from selenium import webdriver
from selenium.webdriver.common.by import By 
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import StaleElementReferenceException
import time
import pandas as pd 
import os 

class RMP_Scraper:
    def __init__(self, driver):
        self.driver = driver
        self.show_more_button = None

    def set_show_more_button(self):
        """Find the Show More button, and attaches an ID to it to find it easily"""
        try:
            Xpath = '//*[@id="root"]/div/div/div[4]/div[1]/div[1]/div[4]/button'
            self.show_more_button = WebDriverWait(self.driver, 10).until(
                EC.visibility_of_element_located((By.XPATH, Xpath)))
            self.driver.execute_script(
                "arguments[0].setAttribute('id', arguments[1]);", self.show_more_button, "RMP_Scrape")
        except Exception as e:
            print(f"Set Show More Button: Some error occurred \n {str(e)}")
            raise e

    def push_show_more_button(self):
        """Push the show_more_button until it's no longer available"""
        if self.show_more_button is None: return ""
        
        try:
            print("Searching for more Professors...")
            self.driver.execute_script("arguments[0].click();", self.show_more_button)
            self.show_more_button = None  # Reset the button reference
            
            # Wait for the "Show More" button to reappear
            try:
                self.set_show_more_button()
            except Exception as e:
                print("No more 'Show More' button found.")
                return ""  # No more "Show More" button, we're done
        except StaleElementReferenceException as ser:
            # The reference to the button is stale, attempt to find it again
            print("Stale reference, finding the button again...")
            self.set_show_more_button()
            self.driver.execute_script("arguments[0].click();", self.show_more_button)
        except Exception as e:
            print(f"Push Show More Button: Some error occurred - {str(e)}")
            return ""
    
    def extract_professors(self):
        """Extract professors data from the visible professor cards"""
        prof_cards = self.driver.find_elements(By.CLASS_NAME, "TeacherCard__StyledTeacherCard-syjs0d-0.dLJIlx")
        prof_info = []
    

        for prof in prof_cards:
            
            try:
                prof_data = {}
                self.driver.execute_script("arguments[0].scrollIntoView();", prof)

                # Extract professor's name
                try:
                    name_element = prof.find_element(By.CLASS_NAME, "CardName__StyledCardName-sc-1gyrgim-0")
                    prof_data['name'] = name_element.text.strip()
                except Exception:
                    prof_data['name'] = "N/A"

                # Extract rating value
                try:
                    rating_element = prof.find_element(By.CLASS_NAME, "CardNumRating__CardNumRatingNumber-sc-17t4b9u-2")
                    prof_data['rating_val'] = rating_element.text.strip()
                except Exception:
                    prof_data['rating_val'] = "N/A"

                # Extract number of ratings
                try:
                    num_ratings_element = prof.find_element(By.CLASS_NAME, "CardNumRating__CardNumRatingCount-sc-17t4b9u-3")
                    prof_data['num_of_ratings'] = num_ratings_element.text.strip()
                except Exception:
                    prof_data['num_of_ratings'] = "N/A"

                # Extract "Would Take Again" percentage
                try:
    # Find the element containing the percentage (e.g., 100%)
                    take_again_element = prof.find_element(By.XPATH, ".//div[contains(@class, 'CardFeedback__CardFeedbackNumber-lq6nix-2') and contains(text(), '%')]")
                    take_again_text = take_again_element.text.strip()
                    prof_data['percent_would_take_again'] = take_again_text

                except Exception:
                    prof_data['percent_would_take_again'] = "N/A"
                    print("Could not find 'Would Take Again' percentage") 

            # Extract "Difficulty Level"
                try:
                    diff_level_text = prof.find_element(By.XPATH, ".//div[contains(@class, 'CardFeedback__CardFeedbackNumber-lq6nix-2 hroXqf')and contains(text(), '.')]").text.strip()
                    prof_data['diff_level'] = diff_level_text
                except Exception:
                    prof_data['diff_level'] = "N/A"

                try:
                    prof_data['href'] = prof.get_attribute("href")
                except Exception:
                    prof_data['href'] = "N/A"

                prof_info.append(prof_data)

            except Exception as e:
                print(f"Error extracting data for professor: {str(e)}")
        
        return prof_info

def main():
    url = 'https://www.ratemyprofessors.com/search/professors/1466?q=*'
    
    # Use Safari WebDriver
    driver = webdriver.Safari()

    driver.get(url)

    # Wait for the page to stop loading (check the "loading" indicator)
    # WebDriverWait(driver, 30).until(
    #     lambda driver: driver.execute_script('return document.readyState') == 'complete' and
    #                    driver.execute_script('return document.querySelector("body").classList.contains("loading")') == False
    # )

    # close the cookies popup 
    close_button = driver.find_element(By.CSS_SELECTOR, "div.FullPageModal__StyledFullPageModal-sc-1tziext-1 button.Buttons__Button-sc-19xdot-1.CCPAModal__StyledCloseButton-sc-10x9kq-2")
    close_button.click()

    # Repeatedly click the "Show More" button until it's gone
    scraper = RMP_Scraper(driver)
    curr_count = 0
    show_more = True
    unique_professors = set()
    all_prof_info = []
    target_professors = 1565
    past_count = -1

    while curr_count != past_count:
        current_professors = scraper.extract_professors()

        # Add current professors to the unique set
        for prof in current_professors:
            if prof['name'] not in unique_professors:
                past_count =curr_count
                unique_professors.add(prof['name'])
                print(prof['name'])
                all_prof_info.append(prof)
                curr_count += 1
                print(curr_count)
                print(past_count)

        try:
            show_more = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, "//button[text()='Show More']"))
            )
            y_position = show_more.location['y'] - 200
            driver.execute_script(f"window.scrollTo(0, {y_position})")
            show_more.click()
            time.sleep(1.5)
        except Exception as e:
            print("No more 'Show More' button found.")
            break

    df = pd.DataFrame(all_prof_info, columns=['name', 'rating_val', 'num_of_ratings', 'percent_would_take_again', 'diff_level', 'href'])
    output_path = 'prof_info.csv'

    df.to_csv(output_path, mode='a', header=not os.path.exists(output_path))

    prof_hrefs = [prof['href'] for prof in all_prof_info]

    with open("teacher_links.txt", "w") as f:
        for link in prof_hrefs:
            f.write(link + "\n")

    print(f"Saved {len(prof_hrefs)} links.")

    time.sleep(5)  # Allow time to see the action if needed

    driver.quit()  # Close the browser when done

if __name__ == "__main__":
    main()
