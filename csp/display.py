"""
Display output for CSP in a simple GUI
page for full schedule,
pages for each day of week
page for hypothetical schedule for a student

"""


import tkinter as tk
from tkinter import ttk, scrolledtext
from csp import Course, ScheduleAssignment, DAYS
from typing import List, Dict
import random


def display_solution_gui(courses: List[Course], solution: Dict[str, ScheduleAssignment]):
    # Display a simple popup if no solution was found
    if solution is None:
        root = tk.Tk()
        root.title("Course Schedule")
        tk.Label(root, text="No solution found!", font=("Arial", 12)).pack(padx=20, pady=20)
        root.mainloop()
        return

    # Format the complete weekly schedule grouped by department and year
    def format_full_schedule():
        lines = ["Full Course Schedule\n", "=" * 80 + "\n"]
        by_dept_year = {}
        for course in courses:
            key = (course.department, course.year_level)
            by_dept_year.setdefault(key, []).append((course, solution[course.code]))
        for (dept, year), assignments in sorted(by_dept_year.items()):
            lines.append(f"{dept} Year {year} Courses:")
            lines.append("-" * 80)
            for course, assignment in sorted(assignments, key=lambda x: x[0].code):
                lines.append(f"{course.code}: {course.name}")
                lines.append(f"  Professor: {course.professor}")
                lines.append(f"  Students: {course.num_students}")
                lines.append(f"  Schedule: {', '.join(str(ts) for ts in assignment.time_slots)}")
                lines.append(f"  Room: {assignment.room.name} (Capacity: {assignment.room.capacity})")
                lines.append("")
            lines.append("")
        return "\n".join(lines)

    # Format the schedule for a specific weekday
    def format_day_schedule(day):
        lines = [f"Schedule for {day}:\n", "=" * 80 + "\n"]
        day_courses = []
        for course in courses:
            assignment = solution[course.code]
            day_slots = [ts for ts in assignment.time_slots if ts.day == day]
            if day_slots:
                day_courses.append((course, assignment, day_slots[0]))
        # Sort courses by time
        day_courses.sort(key=lambda x: (x[2].start_hour, x[2].start_minute))
        for course, assignment, ts in day_courses:
            lines.append(f"{ts.start_hour:02d}:{ts.start_minute:02d} - {course.code}: {course.name}")
            lines.append(f"  Professor: {course.professor}")
            lines.append(f"  Room: {assignment.room.name}")
            lines.append("")
        return "\n".join(lines)

    # Generate a random hypothetical student schedule with 5 courses from the same year
    def format_hypothetical_schedule():
        lines = ["Hypothetical Weekly Schedule\n", "=" * 80 + "\n"]
        valid_years = list(set(c.year_level for c in courses))
        if not valid_years:
            return "No valid year levels found."

        # Try repeatedly to generate a valid schedule
        max_attempts = 50
        selected = None

        for _ in range(max_attempts):
            year = random.choice(valid_years)
            year_courses = [c for c in courses if c.year_level == year and c.code in solution]
            if len(year_courses) < 5:
                continue

            depts = list(set(c.department for c in year_courses))
            random.shuffle(depts)

            for dept in depts:
                main_dept_courses = [c for c in year_courses if c.department == dept]
                other_courses = [c for c in year_courses if c.department != dept]
                if len(main_dept_courses) >= 3 and len(other_courses) >= 2:
                    selected = random.sample(main_dept_courses, 3) + random.sample(other_courses, 2)
                    break

            if selected:
                break

        # Final fallback if no valid combination found
        if not selected:
            return "Could not generate a valid hypothetical schedule after multiple attempts."


        # Organize selected course sessions into a weekly layout
        schedule = {day: [] for day in DAYS}
        for course in selected:
            for ts in solution[course.code].time_slots:
                schedule[ts.day].append((ts.start_hour, ts.start_minute, course))

        for day in DAYS:
            lines.append(f"\n{day}:")
            lines.append("-" * 40)
            entries = sorted(schedule[day], key=lambda x: (x[0], x[1]))
            for hour, minute, course in entries:
                room_name = solution[course.code].room.name
                lines.append(f"{hour:02d}:{minute:02d} - {course.code}: {course.name} ({course.professor})")
                lines.append(f"  Room: {room_name}")
        return "\n".join(lines)


    # Update displayed schedule based on dropdown selection
    def update_display(*_):
        selected = view_option.get()
        if selected == "Full Schedule":
            text_area.config(state="normal")
            text_area.delete(1.0, tk.END)
            text_area.insert(tk.END, format_full_schedule())
            text_area.config(state="disabled")
        elif selected in DAYS:
            text_area.config(state="normal")
            text_area.delete(1.0, tk.END)
            text_area.insert(tk.END, format_day_schedule(selected))
            text_area.config(state="disabled")
        elif selected == "Hypothetical Schedule":
            text_area.config(state="normal")
            text_area.delete(1.0, tk.END)
            text_area.insert(tk.END, format_hypothetical_schedule())
            text_area.config(state="disabled")

    # Create main GUI window
    root = tk.Tk()
    root.title("Course Schedule Viewer")
    root.geometry("900x600")

    # Dropdown to choose schedule view
    view_option = tk.StringVar(value="Full Schedule")
    options = ["Full Schedule"] + DAYS + ["Hypothetical Schedule"]
    dropdown = ttk.Combobox(root, textvariable=view_option, values=options, state="readonly")
    dropdown.pack(padx=10, pady=10)
    dropdown.bind("<<ComboboxSelected>>", update_display)

    # Scrollable text area to display the selected schedule
    text_area = scrolledtext.ScrolledText(root, wrap=tk.WORD, font=("Courier", 10))
    text_area.pack(expand=True, fill='both')
    update_display()

    root.mainloop()
