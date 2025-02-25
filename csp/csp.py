# https://manningbooks.medium.com/constraint-satisfaction-problems-in-python-a1b4ba8dd3bb

from ortools.sat.python import cp_model
import re
from typing import List, Dict, Tuple, Optional, Set, NamedTuple
from dataclasses import dataclass, field
import time
import itertools

# ROOM TYPES
ROOM_TYPE_AUD = "Aud"          # Auditorium
ROOM_TYPE_AL = "AL"            # Active Learning
ROOM_TYPE_TIERED = "Tiered"    # Tiered classroom
ROOM_TYPE_FLAT = "Flat"        # Flat classroom

# DAYS OF THE WEEK
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri"]

# TIME SLOTS - 1 hour blocks from 8:00 to 17:00
TIME_BLOCKS = [
    (8, 0),   # 8:00 AM
    (9, 0),   # 9:00 AM
    (10, 0),  # 10:00 AM
    (11, 0),  # 11:00 AM
    (12, 0),  # 12:00 PM
    (13, 0),  # 1:00 PM
    (14, 0),  # 2:00 PM
    (15, 0),  # 3:00 PM
    (16, 0),  # 4:00 PM
]

# Course class to represent a course with its properties
@dataclass
class Course:
    code: str
    name: str
    enrollment: int
    professor: str
    course_type: str = "Tiered"  # Default room type needed
    num_sessions: int = 3        # Number of class sessions per week (default: 3)
    
    @property
    def year_level(self) -> int:
        # Extract year level from course code (e.g., PHYS100 -> 1, PHYS200 -> 2)
        match = re.search(r'(\d{3})', self.code)
        if match:
            return int(match.group(1)[0])
        return 0
    
    @property
    def department(self) -> str:
        # Extract department from course code (e.g., PHYS100 -> PHYS)
        match = re.search(r'([A-Z]+)', self.code)
        if match:
            return match.group(1)
        return ""

# Room class to represent a room with its properties
@dataclass
class Room:
    name: str
    capacity: int
    room_type: str  # "Aud", "AL", "Tiered", or "Flat"

# Time slot representation
class TimeSlot(NamedTuple):
    day: str
    start_hour: int
    start_minute: int
    
    def __str__(self) -> str:
        return f"{self.day} {self.start_hour:02d}:{self.start_minute:02d}"
    
    @property
    def time_value(self) -> float:
        """Convert to a float value for easier comparison"""
        return self.start_hour + (self.start_minute / 60)

# Schedule Assignment class
@dataclass
class ScheduleAssignment:
    room: Room
    time_slots: List[TimeSlot]
    
    def __str__(self) -> str:
        room_str = f"{self.room.name} (Capacity: {self.room.capacity}, Type: {self.room.room_type})"
        times_str = ", ".join(str(ts) for ts in self.time_slots)
        return f"Room: {room_str}, Times: {times_str}"

class CourseScheduler:
    def __init__(self, courses: List[Course], rooms: List[Room]):
        self.courses = courses
        self.rooms = rooms
        self.model = cp_model.CpModel()
        self.solver = cp_model.CpSolver()
        
        # Generate all possible time slots
        self.all_time_slots = self._generate_time_slots()
        self.time_slot_lookup = {(ts.day, ts.start_hour, ts.start_minute): i 
                                for i, ts in enumerate(self.all_time_slots)}
        
        # Create decision variables
        self.room_assignments = {}  # {course_code: room_var}
        self.time_block_assignments = {}  # {course_code: time_block_var}
        self.day_assignments = {}  # {course_code: {day: binary_var}}
        
        # Track slot assignments for debugging
        self.slot_assignments = {}  # {course_code: {slot_idx: binary_var}}
        
        # Time preferences
        self.professor_time_preferences = {}  # {professor: preferred_time_blocks}
        self.professor_day_preferences = {}  # {professor: preferred_days}
        
        # Solution callback will be set in build_model
        self.solution_printer = None
    
    def _generate_time_slots(self) -> List[TimeSlot]:
        """Generate all possible time slots for scheduling"""
        time_slots = []
        for day in DAYS:
            for hour, minute in TIME_BLOCKS:
                time_slots.append(TimeSlot(day, hour, minute))
        return time_slots
    
    def add_professor_time_preference(self, professor: str, preferred_times: List[Tuple[int, int]]):
        """Add time-of-day preferences for a professor (hour, minute)"""
        self.professor_time_preferences[professor] = preferred_times
    
    def add_professor_day_preference(self, professor: str, preferred_days: List[str]):
        """Add day preferences for a professor (e.g., ['Mon', 'Tue', 'Wed'])"""
        self.professor_day_preferences[professor] = preferred_days
    
    def build_model(self):
        # Create variables for room and time slot assignments
        for course in self.courses:
            # Get valid rooms for this course
            valid_rooms = [i for i, room in enumerate(self.rooms) 
                          if room.capacity >= course.enrollment and room.room_type == course.course_type]
            
            if not valid_rooms:
                raise ValueError(f"No suitable room found for {course.code}: {course.name} "
                               f"(needs {course.course_type} room with capacity >= {course.enrollment})")
            
            # Create room assignment variable
            self.room_assignments[course.code] = self.model.NewIntVarFromDomain(
                cp_model.Domain.FromValues(valid_rooms),
                f'room_for_{course.code}'
            )
            
            # Create time block assignment variable (which hour of the day)
            self.time_block_assignments[course.code] = self.model.NewIntVar(
                0, len(TIME_BLOCKS) - 1,
                f'time_block_for_{course.code}'
            )
            
            # Create day assignment variables (which days of the week)
            self.day_assignments[course.code] = {}
            for day in DAYS:
                self.day_assignments[course.code][day] = self.model.NewBoolVar(
                    f'day_{day}_for_{course.code}'
                )
            
            # Track slot assignments for each course
            self.slot_assignments[course.code] = {}
            for i, slot in enumerate(self.all_time_slots):
                self.slot_assignments[course.code][i] = self.model.NewBoolVar(
                    f'slot_{i}_for_{course.code}'
                )
                
                # Link slot assignments to day and time block variables
                # If this slot is assigned, then its day must be assigned
                day_var = self.day_assignments[course.code][slot.day]
                slot_var = self.slot_assignments[course.code][i]
                
                # NOT(slot_var) OR day_var
                self.model.AddBoolOr([slot_var.Not(), day_var])
                
                # Find the time block index for this slot
                for block_idx, (hour, minute) in enumerate(TIME_BLOCKS):
                    if slot.start_hour == hour and slot.start_minute == minute:
                        # If slot is assigned, then time block must match
                        time_block_var = self.time_block_assignments[course.code]
                        
                        # Create helper variable for the equality
                        time_block_matches = self.model.NewBoolVar(f'time_block_matches_{course.code}_{i}')
                        self.model.Add(time_block_var == block_idx).OnlyEnforceIf(time_block_matches)
                        self.model.Add(time_block_var != block_idx).OnlyEnforceIf(time_block_matches.Not())
                        
                        # NOT(slot_var) OR time_block_matches
                        self.model.AddBoolOr([slot_var.Not(), time_block_matches])
            
            # Constraint: Each course must have exactly num_sessions slots assigned
            self.model.Add(sum(self.slot_assignments[course.code].values()) == course.num_sessions)
            
            # Constraint: Each course must use exactly num_sessions different days
            self.model.Add(sum(self.day_assignments[course.code].values()) == course.num_sessions)
            
            # Constraint: All sessions must be at the same time of day (but on different days)
            # This is enforced by the implications above
        
        # Add constraints
        self._add_room_time_overlap_constraints()
        self._add_department_year_constraints()
        self._add_professor_preference_constraints()
        
        # Set up solution printer
        self.solution_printer = SolutionPrinter(
            self.room_assignments,
            self.time_block_assignments,
            self.day_assignments,
            self.slot_assignments,
            self.courses,
            self.rooms,
            self.all_time_slots
        )
    
    def _add_room_time_overlap_constraints(self):
        """No room can be assigned to overlapping time slots"""
        # For each pair of courses
        for i, course1 in enumerate(self.courses):
            for course2 in self.courses[i+1:]:
                # For each pair of time slots
                for slot1_idx, slot1_var in self.slot_assignments[course1.code].items():
                    for slot2_idx, slot2_var in self.slot_assignments[course2.code].items():
                        slot1 = self.all_time_slots[slot1_idx]
                        slot2 = self.all_time_slots[slot2_idx]
                        
                        # If same day and same time block
                        if slot1.day == slot2.day and slot1.start_hour == slot2.start_hour and slot1.start_minute == slot2.start_minute:
                            # Create helper variable for room equality check
                            same_room = self.model.NewBoolVar(f'same_room_{course1.code}_{course2.code}_{slot1_idx}_{slot2_idx}')
                            self.model.Add(self.room_assignments[course1.code] == self.room_assignments[course2.code]).OnlyEnforceIf(same_room)
                            self.model.Add(self.room_assignments[course1.code] != self.room_assignments[course2.code]).OnlyEnforceIf(same_room.Not())
                            
                            # If both slots are assigned AND they use the same room, then it's a conflict
                            # NOT(slot1 AND slot2 AND same_room)
                            self.model.AddBoolOr([
                                slot1_var.Not(),
                                slot2_var.Not(),
                                same_room.Not()
                            ])
    
    def _add_department_year_constraints(self):
        """No same-department, same-year courses should have overlapping time slots"""
        # Group courses by department and year
        dept_year_groups = {}
        for course in self.courses:
            key = (course.department, course.year_level)
            if key not in dept_year_groups:
                dept_year_groups[key] = []
            dept_year_groups[key].append(course)
        
        # For each group, ensure no overlapping time slots
        for (dept, year), group in dept_year_groups.items():
            for i, course1 in enumerate(group):
                for course2 in group[i+1:]:
                    # For each pair of time slots
                    for slot1_idx, slot1_var in self.slot_assignments[course1.code].items():
                        for slot2_idx, slot2_var in self.slot_assignments[course2.code].items():
                            slot1 = self.all_time_slots[slot1_idx]
                            slot2 = self.all_time_slots[slot2_idx]
                            
                            # If same day and same time block
                            if slot1.day == slot2.day and slot1.start_hour == slot2.start_hour and slot1.start_minute == slot2.start_minute:
                                # Add constraint: NOT(slot1 AND slot2)
                                self.model.AddBoolOr([slot1_var.Not(), slot2_var.Not()])
    
    def _add_professor_preference_constraints(self):
        """Add constraints for professor preferences"""
        # Add time-of-day preferences
        for professor, preferred_times in self.professor_time_preferences.items():
            # Find all courses taught by this professor
            prof_courses = [c for c in self.courses if c.professor == professor]
            
            if not prof_courses:
                continue
            
            # Convert preferred times to time block indices
            preferred_blocks = []
            for hour, minute in preferred_times:
                for block_idx, (block_hour, block_minute) in enumerate(TIME_BLOCKS):
                    if block_hour == hour and block_minute == minute:
                        preferred_blocks.append(block_idx)
            
            if not preferred_blocks:
                continue
            
            # Add constraints for each course
            for course in prof_courses:
                # Create a constraint that the time block must be one of the preferred ones
                self.model.AddAllowedAssignments(
                    [self.time_block_assignments[course.code]], 
                    [(block_idx,) for block_idx in preferred_blocks]
                )
        
        # Add day-of-week preferences
        for professor, preferred_days in self.professor_day_preferences.items():
            # Find all courses taught by this professor
            prof_courses = [c for c in self.courses if c.professor == professor]
            
            if not prof_courses:
                continue
            
            # Add constraints for each course
            for course in prof_courses:
                # For each day, if it's not in preferred days, forbid it
                for day in DAYS:
                    if day not in preferred_days:
                        self.model.Add(self.day_assignments[course.code][day] == 0)
    
    def add_professor_room_preference(self, course_code: str, preferred_rooms: List[str]):
        """Add constraint for professor's room preference"""
        if course_code not in [c.code for c in self.courses]:
            raise ValueError(f"Course {course_code} not found")
        
        # Convert room names to indices
        preferred_indices = [i for i, room in enumerate(self.rooms) if room.name in preferred_rooms]
        
        if not preferred_indices:
            raise ValueError(f"None of the preferred rooms are valid for {course_code}")
        
        # Create a constraint that the room assignment must be one of the preferred rooms
        self.model.AddAllowedAssignments(
            [self.room_assignments[course_code]], 
            [(i,) for i in preferred_indices]
        )
    
    def solve(self, time_limit_seconds: int = 60) -> Optional[Dict[str, ScheduleAssignment]]:
        """Solve the course scheduling problem"""
        start_time = time.time()
        
        # Set time limit
        self.solver.parameters.max_time_in_seconds = time_limit_seconds
        
        # Solve
        status = self.solver.Solve(self.model, self.solution_printer)
        
        end_time = time.time()
        solve_time = end_time - start_time
        
        # Check status
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            print(f"Solution found in {solve_time:.2f} seconds")
            return self._extract_solution()
        elif status == cp_model.INFEASIBLE:
            print("The problem is infeasible - no solution exists that satisfies all constraints")
            return None
        else:
            print(f"Solver stopped with status: {status}")
            return None
    
    def _extract_solution(self) -> Dict[str, ScheduleAssignment]:
        """Extract solution from the solver"""
        solution = {}
        for course in self.courses:
            # Get assigned room
            room_index = self.solver.Value(self.room_assignments[course.code])
            room = self.rooms[room_index]
            
            # Get assigned time slots
            time_slots = []
            for slot_idx, slot_var in self.slot_assignments[course.code].items():
                if self.solver.Value(slot_var) == 1:
                    time_slots.append(self.all_time_slots[slot_idx])
            
            # Sort time slots by day order
            day_order = {day: i for i, day in enumerate(DAYS)}
            time_slots.sort(key=lambda ts: (day_order[ts.day], ts.start_hour, ts.start_minute))
            
            # Create schedule assignment
            solution[course.code] = ScheduleAssignment(room, time_slots)
        
        return solution


class SolutionPrinter(cp_model.CpSolverSolutionCallback):
    """Solution callback to print intermediate solutions"""
    
    def __init__(self, room_assignments, time_block_assignments, day_assignments, slot_assignments, 
                 courses, rooms, all_time_slots):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.room_assignments = room_assignments
        self.time_block_assignments = time_block_assignments
        self.day_assignments = day_assignments
        self.slot_assignments = slot_assignments
        self.courses = courses
        self.rooms = rooms
        self.all_time_slots = all_time_slots
        self.solution_count = 0
        self.solution_limit = 5  # Limit the number of solutions to print
    
    def on_solution_callback(self):
        self.solution_count += 1
        if self.solution_count <= self.solution_limit:
            print(f"\nSolution {self.solution_count}:")
            print("=" * 80)
            
            for course in self.courses:
                # Get assigned room
                room_index = self.Value(self.room_assignments[course.code])
                room = self.rooms[room_index]
                
                # Get assigned time block
                time_block_index = self.Value(self.time_block_assignments[course.code])
                time_block = TIME_BLOCKS[time_block_index]
                
                # Get assigned days
                assigned_days = []
                for day, day_var in self.day_assignments[course.code].items():
                    if self.Value(day_var) == 1:
                        assigned_days.append(day)
                
                # Get time slots
                time_slots = []
                for slot_idx, slot_var in self.slot_assignments[course.code].items():
                    if self.Value(slot_var) == 1:
                        time_slots.append(self.all_time_slots[slot_idx])
                
                # Sort time slots by day
                day_order = {day: i for i, day in enumerate(DAYS)}
                time_slots.sort(key=lambda ts: day_order[ts.day])
                
                print(f"{course.code}: {course.name}")
                print(f"  Professor: {course.professor}")
                print(f"  Enrollment: {course.enrollment}")
                print(f"  Days: {', '.join(assigned_days)}")
                print(f"  Time: {time_block[0]:02d}:{time_block[1]:02d}")
                print(f"  Schedule: {', '.join(str(ts) for ts in time_slots)}")
                print(f"  Room: {room.name} (Capacity: {room.capacity}, Type: {room.room_type})")
                print()
        
        if self.solution_count >= self.solution_limit:
            print(f"Solution limit ({self.solution_limit}) reached. Stopping search...")
            self.StopSearch()


def print_final_solution(courses: List[Course], solution: Dict[str, ScheduleAssignment]):
    """Print the final solution in a nice format"""
    if solution is None:
        print("No solution found!")
        return
    
    print("\nFinal Schedule Solution:")
    print("=" * 80)
    
    # Group by department and year for more organized output
    by_dept_year = {}
    for course in courses:
        key = (course.department, course.year_level)
        if key not in by_dept_year:
            by_dept_year[key] = []
        by_dept_year[key].append((course, solution[course.code]))
    
    # Print by department and year
    for (dept, year), assignments in sorted(by_dept_year.items()):
        print(f"\n{dept} Year {year} Courses:")
        print("-" * 80)
        for course, assignment in sorted(assignments, key=lambda x: x[0].code):
            print(f"{course.code}: {course.name}")
            print(f"  Professor: {course.professor}")
            print(f"  Enrollment: {course.enrollment}")
            print(f"  Schedule: {', '.join(str(ts) for ts in assignment.time_slots)}")
            print(f"  Room: {assignment.room.name} (Capacity: {assignment.room.capacity}, Type: {assignment.room.room_type})")
            print()

# Print a schedule for a specific day
def print_day_schedule(day: str, courses: List[Course], solution: Dict[str, ScheduleAssignment]):
    """Print the schedule for a specific day of the week"""
    if solution is None:
        print("No solution found!")
        return
    
    print(f"\nSchedule for {day}:")
    print("=" * 80)
    
    # Find all courses that have sessions on this day
    day_courses = []
    for course in courses:
        assignment = solution[course.code]
        # Check if any time slot is on the requested day
        day_slots = [ts for ts in assignment.time_slots if ts.day == day]
        if day_slots:
            day_courses.append((course, assignment, day_slots[0]))
    
    # Sort by time
    day_courses.sort(key=lambda x: (x[2].start_hour, x[2].start_minute))
    
    # Print schedule
    for course, assignment, time_slot in day_courses:
        print(f"{time_slot.start_hour:02d}:{time_slot.start_minute:02d} - ", end="")
        print(f"{course.code}: {course.name}")
        print(f"  Professor: {course.professor}")
        print(f"  Room: {assignment.room.name}")
        print()


# Example usage
def run_example():
    # Define rooms with the new room types
    rooms = [
        Room("A101", 30, ROOM_TYPE_TIERED),
        Room("A102", 50, ROOM_TYPE_TIERED),
        Room("B201", 100, ROOM_TYPE_TIERED),
        Room("B202", 150, ROOM_TYPE_TIERED),
        Room("C301", 200, ROOM_TYPE_AUD),
        Room("C302", 180, ROOM_TYPE_AUD),
        Room("D401", 35, ROOM_TYPE_AL),
        Room("D402", 40, ROOM_TYPE_AL),
        Room("D403", 30, ROOM_TYPE_AL),
        Room("E501", 40, ROOM_TYPE_FLAT),
        Room("E502", 60, ROOM_TYPE_FLAT),
    ]
    
    # Define courses with their properties (but no fixed days or times)
    courses = [
        # PHYS courses
        Course("PHYS101", "Intro to Physics", 45, "Dr. Smith", ROOM_TYPE_TIERED, 3),
        Course("PHYS102", "Mechanics", 30, "Dr. Johnson", ROOM_TYPE_AL, 2),
        Course("PHYS110", "Physics Lab", 20, "Dr. Brown", ROOM_TYPE_AL, 1),
        
        # MATH courses
        Course("MATH101", "Calculus I", 50, "Dr. Davis", ROOM_TYPE_TIERED, 3),
        Course("MATH102", "Linear Algebra", 40, "Dr. Taylor", ROOM_TYPE_TIERED, 2),
        
        # COMP courses
        Course("COMP101", "Intro to Programming", 60, "Dr. Anderson", ROOM_TYPE_FLAT, 3),
        Course("COMP110", "Programming Lab", 25, "Dr. Thomas", ROOM_TYPE_AL, 2),
    ]
    
    # Create and solve the scheduling problem
    scheduler = CourseScheduler(courses, rooms)
    
    # Add professor time preferences (morning hours only)
    scheduler.add_professor_time_preference("Dr. Smith", [
        (9, 0),  # 9:00 AM
    ])
    
    # Add professor time preferences (afternoon hours only)
    scheduler.add_professor_time_preference("Dr. Brown", [
        (13, 0),  # 1:00 PM
        (14, 0),  # 2:00 PM
    ])
    
    # Add professor day preferences
    scheduler.add_professor_day_preference("Dr. Smith", ["Mon", "Tue", "Wed"])
    scheduler.add_professor_day_preference("Dr. Johnson", ["Mon", "Wed", "Thu"])
    scheduler.add_professor_day_preference("Dr. Brown", ["Tue", "Thu"])
    
    # Build the model
    scheduler.build_model()
    
    # Add room preferences
    scheduler.add_professor_room_preference("PHYS101", ["B201", "B202"])
    
    # Solve
    solution = scheduler.solve(time_limit_seconds=60)
    
    # Print final solution
    print_final_solution(courses, solution)
    
    # Print schedule for Monday
    print_day_schedule("Mon", courses, solution)


if __name__ == "__main__":
    run_example()