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
        """Find the Show More button, and attach an ID to it to find it easily"""
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

            # Add scroll to ensure more results are loaded
            time.sleep(3)  # Wait for loading
            self.driver.execute_script("window.scrollBy(0, 1000);")  # Adjust scroll height if needed

        except StaleElementReferenceException as ser:
            # The reference to the button is stale, attempt to find it again
            print("Stale reference, finding the button again...")
            self.set_show_more_button()
            self.driver.execute_script("arguments[0].click();", self.show_more_button)
        except Exception as e:
            print(f"Push Show More Button: Some error occurred - {str(e)}")
            return ""

def main():
    url = 'https://www.ratemyprofessors.com/search/professors/1466?q=*'
    
    # Use Safari WebDriver
    driver = webdriver.Safari()

    driver.get(url)

    # Wait for the page to stop loading (check the "loading" indicator)
    WebDriverWait(driver, 30).until(
        lambda driver: driver.execute_script('return document.readyState') == 'complete' and
                       driver.execute_script('return document.querySelector("body").classList.contains("loading")') == False
    )

    # Close the cookies popup
    close_button = driver.find_element(By.CSS_SELECTOR, "div.FullPageModal__StyledFullPageModal-sc-1tziext-1 button.Buttons__Button-sc-19xdot-1.CCPAModal__StyledCloseButton-sc-10x9kq-2")
    close_button.click()

    # Initialize the scraper
    scraper = RMP_Scraper(driver)
    scraper.set_show_more_button()

    # Number of professors you want to scrape
    target_professors = 1565
    count = 0

    all_prof_info = []

    while count < target_professors:
        count += 1
        print(f"Loading professor {count}/{target_professors}")

        # Locate the "Show More" button and click it
        show_more = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, "//button[text()='Show More']"))
        )
        y_position = show_more.location['y'] - 200
        driver.execute_script(f"window.scrollTo(0, {y_position})")
        show_more.click()

        # Wait for new content to load
        time.sleep(3)  # Adjust time as needed

        # Extract professor cards after clicking "Show More"
        prof_cards = driver.find_elements(By.CLASS_NAME, "TeacherCard__StyledTeacherCard-syjs0d-0.dLJIlx")
        
        for prof in prof_cards:
            try:
                prof_info = []
                driver.execute_script("arguments[0].scrollIntoView();", prof)

                # Extract professor's name
                try:
                    name_element = prof.find_element(By.CLASS_NAME, "CardName__StyledCardName-sc-1gyrgim-0")
                    name = name_element.text.strip()
                except Exception:
                    name = "N/A"
                    print(f"❌ Name not found for professor {count}")

                # Extract rating value
                try:
                    rating_element = prof.find_element(By.CLASS_NAME, "CardNumRating__CardNumRatingNumber-sc-17t4b9u-2")
                    rating_val = rating_element.text.strip()
                except Exception:
                    rating_val = "N/A"
                    print(f"❌ Rating value not found for professor {count}")

                # Extract number of ratings
                try:
                    num_ratings_element = prof.find_element(By.CLASS_NAME, "CardNumRating__CardNumRatingCount-sc-17t4b9u-3")
                    num_ratings = num_ratings_element.text.strip()
                except Exception:
                    num_ratings = "N/A"
                    print(f"❌ Number of ratings not found for professor {count}")

                # Extract "Would Take Again" percentage
                try:
                    take_again_element = prof.find_element(By.XPATH, "(//div[@class='CardFeedback__CardFeedbackNumber-lq6nix-2 hroXqf'])[1]")
                    take_again = take_again_element.text.strip()
                except Exception:
                    take_again = "N/A"
                    print(f"❌ 'Would Take Again' percentage not found for professor {count}")

                # Extract "Difficulty Level"
                try:
                    diff_level_element = prof.find_element(By.XPATH, "(//div[@class='CardFeedback__CardFeedbackNumber-lq6nix-2 hroXqf'])[2]")
                    diff_level = diff_level_element.text.strip()
                except Exception:
                    diff_level = "N/A"
                    print(f"❌ Difficulty level not found for professor {count}")

                # Store extracted data
                all_prof_info.append([name, rating_val, num_ratings, take_again, diff_level])

            except Exception as e:
                print(f"Error extracting data for professor {count}: {str(e)}")

    # Save the data to a CSV file
    df = pd.DataFrame(all_prof_info, columns=['name', 'rating_val', 'num_of_ratings', 'percent_would_take_again', 'diff_level'])
    output_path = 'prof_info.csv'
    df.to_csv(output_path, mode='a', header=not os.path.exists(output_path))

    # Extract and save professor links
    prof_cards = driver.find_elements(By.CLASS_NAME, "TeacherCard__StyledTeacherCard-syjs0d-0.dLJIlx")
    prof_hrefs = [a.get_attribute("href") for a in prof_cards if a.get_attribute("href")]

    with open("teacher_links.txt", "w") as f:
        for link in prof_hrefs:
            f.write(link + "\n")

    print(f"Saved {len(prof_hrefs)} links.")

    # Close the browser when done
    time.sleep(5)  # Allow time to see the action if needed
    driver.quit()  # Close the browser

if __name__ == "__main__":
    main()
