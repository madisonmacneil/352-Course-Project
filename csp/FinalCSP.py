"""
Course Scheduling with Real Data

This script loads data from CSV files and uses it with the constraint satisfaction
problem solver to generate a course schedule.
"""

import pandas as pd
import time
from typing import List, Dict
import sys
import random
from display import display_solution_gui

random.seed(2)

# Import CSP classes
from csp import (
    Course, Room, CourseScheduler, 
    print_final_solution, print_day_schedule,
    TIME_BLOCKS, DAYS
)

# Function to load professor preferences
def load_professor_preferences(scheduler, preferences_csv_path):
    """
    Load professor preferences from CSV and apply them to the scheduler.
    
    Args:
        scheduler: CourseScheduler instance
        preferences_csv_path: Path to the preferences CSV file
    """
    try:
        prefs_df = pd.read_csv(preferences_csv_path)
        
        # Group preferences by professor
        prof_count = 0
        
        for prof, group in prefs_df.groupby('prof'):
            exclusions = []
            # All rows for this professor should have the same weight
            weight = group.iloc[0]['weight']
            
            for _, row in group.iterrows():
                hour = int(row['hour'])
                minute = int(row['minute'])
                days = row['days'].split(',')
                # Sort days to maintain Mon-Fri order
                days.sort(key=lambda day: DAYS.index(day))
                
                exclusions.append((hour, minute, days))
            
            # Add all exclusions for this professor at once
            if exclusions:  # Only add if there are exclusions
                scheduler.add_professor_time_exclusion(
                    prof,
                    exclusions,
                    weight=weight
                )
                prof_count += 1
        
        print(f"Loaded preferences for {prof_count} professors from {preferences_csv_path}")
        
    except Exception as e:
        print(f"Error loading professor preferences: {e}")
        return False
    
    return True

def main():
    courses_csv = '352-Course-Project/csp/course_csp_limited.csv'
    rooms_csv = '352-Course-Project/csp/rooms.csv'
    preferences_csv = '352-Course-Project/csp/prof_preferences.csv'
    
    # Load rooms
    try:
        print(f"Loading rooms from {rooms_csv}...")
        rooms_df = pd.read_csv(rooms_csv)
        rooms = [Room(row['room_name'], row['capacity']) for _, row in rooms_df.iterrows()]
        print(f"Loaded {len(rooms)} rooms")
    except Exception as e:
        print(f"Error loading rooms: {e}")
        sys.exit(1)
    
    # Load courses
    try:
        print(f"Loading courses from {courses_csv}...")
        courses_df = pd.read_csv(courses_csv)
        courses = [
            Course(
                row['course_code'], 
                row['name'], 
                row['num_students'], 
                row['prof']
            ) for _, row in courses_df.iterrows()
        ]
        print(f"Loaded {len(courses) // 2} courses")
    except Exception as e:
        print(f"Error loading courses: {e}")
        sys.exit(1)
    
    # Only look at half the courses - assume half of them run per semester anyway - looking at all 642 took way too long 
    # courses = random.sample(courses, len(courses) // 2 ) 
    courses = random.sample(courses, 200)
    
    print("Creating scheduler...")
    scheduler = CourseScheduler(courses, rooms)
    
    # Load professor preferences
    print(f"Loading professor preferences from {preferences_csv}...")
    load_professor_preferences(scheduler, preferences_csv)
    
    # Build the model
    print("Building scheduling model...")
    try:
        scheduler.build_model()
    except ValueError as e:
        print(f"Error building model: {e}")
        sys.exit(1)
    
    # Solve the model
    print("Solving scheduling problem...")
    start_time = time.time()
    time_limit = 1500 
    solution = scheduler.solve(time_limit_seconds=time_limit)
    end_time = time.time()
    print(f"Solving took {end_time - start_time:.2f} seconds")
    
    if solution:
        print_final_solution(courses, solution)
        for day in DAYS:
            print_day_schedule(day, courses, solution)
    else:
        if scheduler.timed_out:
            print("No solution was found because the solver ran out of time.")
        else:
            print("No solution was found. The problem might be infeasible. Try adjusting constraints or increasing the time limit.")

    display_solution_gui(courses, solution)  # Call this after solving the CSP


if __name__ == "__main__":
    main()
