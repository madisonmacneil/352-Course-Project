# Create prof preferences CSV, profs would all have opportnity to input preferences - to simulate this we will just generate some random preferences.

import pandas as pd
import random

random.seed(1)

TIME_BLOCKS = [
    (8, 30),   # 8:30 AM
    (9, 30),   # 9:30 AM
    (10, 30),  # 10:30 AM
    (11, 30),  # 11:30 AM
    (12, 30),  # 12:30 PM
    (13, 30),  # 1:30 PM
    (14, 30),  # 2:30 PM
    (15, 30),  # 3:30 PM
    (16, 30),  # 4:30 PM
]

DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri"]

# Load the courses CSV to get professors
courses_df = pd.read_csv('352-Course-Project/csp/course_csp.csv')
# Get unique professor names - some profs teach more than one course
professors = courses_df['prof'].unique()

# Generate preferences data
preferences_data = []

professor_weights = {prof: random.randint(30, 75) for prof in professors} # simulate age, older prof gets higher priority 

for prof in professors:
    # 80% chance of no exclusions - too many prof preferences kept giving no solutions so assume only 20% of profs would have preferences
    # This number allowed for the csp to get feasable solutions more often
    if random.random() < 0.8: 
        continue

    weight = professor_weights[prof]
    
    # Profs whow have preferences can request 1 to 3 seperate times off. so 8:30, 10:30, 1:30 for example 
    num_exclusions = random.randint(1, 3)
    excluded_times = random.sample(TIME_BLOCKS, num_exclusions)
    
    for hour, minute in excluded_times:
        # For each exclusion, select 1-3 days to exclude, so MON TUES off for 8:30, WED, THURS, FRI off at 10:30, etc. 
        num_days = random.randint(1, 3)
        excluded_days = random.sample(DAYS, num_days)
        excluded_days.sort(key=lambda day: DAYS.index(day))
        excluded_days_str = ','.join(excluded_days)
        
        preferences_data.append({
            'prof': prof,
            'hour': hour,
            'minute': minute,
            'days': excluded_days_str,
            'weight': weight
        })

# Create DataFrame and save to CSV
preferences_df = pd.DataFrame(preferences_data)
preferences_df.to_csv('352-Course-Project/csp/prof_preferences.csv', index=False)

print(f"Generated prof_preferences.csv with {len(preferences_df)} preference entries for {len(professors)} professors")

# Print sample of the data
print("\nSample of generated preferences:")
print(preferences_df.head())

print("\nExample of how this would be used in the scheduler:")

# Get the first 2 unique professors in order they appear in the DataFrame
first_two_profs = []
for prof in preferences_df['prof']:
    if prof not in first_two_profs:
        first_two_profs.append(prof)
    if len(first_two_profs) == 2:
        break

# Show examples for these professors
for prof in first_two_profs:
    group = preferences_df[preferences_df['prof'] == prof]
    print(f'scheduler.add_professor_time_exclusion("{prof}", [')
    for _, row in group.iterrows():
        days_list = row['days'].split(',')
        days_str = ', '.join([f'"{day}"' for day in days_list])
        print(f'    ({row["hour"]}, {row["minute"]}, [{days_str}]),')
    print(f'], weight={group.iloc[0]["weight"]})')