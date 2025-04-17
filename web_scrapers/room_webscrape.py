"""
This file is for scraping the necessary data for the CSP from queensu.ca
Specifically getting data on the classroom spaces available on campus,
Other Data such as courses and professors have been scraped for previously
"""

import csv
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup

# Initialize WebDriver
driver = webdriver.Chrome()

# List of classrooms
classrooms = [
    "Biosciences Complex", "Botterell Hall", "Chernoff Hall",
    "Duncan McArthur Hall", "Dunning Hall", "Dupuis Hall",
    "Ellis Hall", "Etherington Hall", "Goodwin Hall",
    "Humphrey Hall", "isabel-bader-centre-performing-arts",
    "Jeffery Hall", "Kinesiology and Health Studies", "Kingston Hall",
    "Law Building", "Mackintosh-Corry Hall", "McLaughlin Hall",
    "Miller Hall", "Nicol Hall", "Ontario Hall",
    "Richardson Lab", "Robert Sutherland Hall", "Stirling Hall",
    "Theological Hall", "Walter Light Hall", "Watson Hall"
]

# Base URL
base_url = "https://www.queensu.ca/classrooms/classrooms/"

# CSV File Path - to data folder
csv_filename = "../data/classrooms.csv"

# Open CSV file for writing
with open(csv_filename, mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)

    # column headers
    writer.writerow(["Building", "Room Number", "Room Type", "Seats"])

    for classroom in classrooms:
        # Convert classroom name to URL-friendly slug
        slug = classroom.lower().replace(' ', '-')
        classroom_url = f"{base_url}{slug}"

        # Visit the page
        driver.get(classroom_url)
        soup = BeautifulSoup(driver.page_source, "html.parser")

        # Find the first table
        first_table = soup.find("table")

        if first_table:
            # Extract table rows
            rows = first_table.find_all("tr")
            
            if rows:
                # Get column names (first row)
                col_names = [col.text.strip() for col in rows[0].find_all(["td", "th"])]

                # Ensure expected format: ['Room Number', 'Room Type', 'Number of Student Seats']
                if col_names == ["Room Number", "Room Type", "Number of Student Seats"]:
                    for row in rows[1:]:  # Skip header row
                        cols = [col.text.strip() for col in row.find_all(["td", "th"])]
                        if len(cols) == 3:
                            writer.writerow([classroom] + cols)  # Prepend building name
                else:
                    print(f"Skipping {classroom}: Unexpected column format")
            else:
                print(f"Skipping {classroom}: No rows found in table")
        else:
            print(f"Skipping {classroom}: No table found")
            
driver.quit()

print(f"Data saved to {csv_filename}")
