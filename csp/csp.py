"""
CSP functions and setup
"""

from ortools.sat.python import cp_model
import re
from typing import List, Dict, Tuple, Optional, Set, NamedTuple
from dataclasses import dataclass, field
import time

# DAYS OF THE WEEK
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri"]

# TIME SLOTS - 1 hour blocks from 8:30am to 4:30pm
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

@dataclass
class Course:
    code: str
    name: str
    num_students: int
    professor: str
    num_sessions: int = 3        # Number of class sessions per week - assume each class runs 3 times a week 
    
    @property
    def year_level(self) -> int:
        # Extract year level from course code (ex., PHYS100 -> 1, PHYS200 -> 2)
        match = re.search(r'(\d{3})', self.code)
        if match:
            return int(match.group(1)[0])
        return 0
    
    @property
    def department(self) -> str:
        # Extract department from course code (ex., PHYS100 -> PHYS)
        match = re.search(r'([A-Z]+)', self.code)
        if match:
            return match.group(1)
        return ""

@dataclass
class Room:
    name: str
    capacity: int

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

@dataclass
class ScheduleAssignment:
    room: Room
    time_slots: List[TimeSlot]
    
    def __str__(self) -> str:
        room_str = f"{self.room.name} (Capacity: {self.room.capacity})"
        times_str = ", ".join(str(ts) for ts in self.time_slots)
        return f"Room: {room_str}, Times: {times_str}"

class CourseScheduler:
    def __init__(self, courses: List[Course], rooms: List[Room]):
        self.courses = courses
        self.rooms = rooms
        self.preference_penalties = []
        self.preference_weights = {}
        self.model = cp_model.CpModel()
        self.solver = cp_model.CpSolver()

        # Generate all possible time slots
        self.all_time_slots = self._generate_time_slots()
        self.time_slot_lookup = {(ts.day, ts.start_hour, ts.start_minute): i 
                                for i, ts in enumerate(self.all_time_slots)}
        
        self.room_assignments = {}  # {course_code: room_var}

        self.time_block_assignments = {}  # {course_code: {day: time_block_var}}
        self.day_assignments = {}  # {course_code: {day: binary_var}}
        
        self.slot_assignments = {}  # {course_code: {slot_idx: binary_var}}
        
        self.professor_time_exclusions = {}  # {professor: excluded_time_blocks}
        self.professor_day_preferences = {}  # {professor: preferred_days}
        
        self.solution_printer = None
    
    def _generate_time_slots(self) -> List[TimeSlot]:
        """Generate all possible time slots for scheduling"""
        time_slots = []
        for day in DAYS:
            for hour, minute in TIME_BLOCKS:
                time_slots.append(TimeSlot(day, hour, minute))
        return time_slots
    
    def add_professor_time_exclusion(self, professor: str, excluded_time_days: List[Tuple[int, int, List[str]]], weight: int = 10):
        """
        Add times when a professor would prefer not to teach (soft constraint)
        for the weight we randomly assign a professor an age to simulate seniority
        Args:
            professor (str): The professor's name
            excluded_time_days (List[Tuple[int, int, List[str]]]): List of (hour, minute, [days]) tuples
            weight (int): Penalty weight for violating this preference (higher = more important) 
        """
        self.professor_time_exclusions[professor] = excluded_time_days
        self.preference_weights[professor] = weight
    
    def build_model(self):
        # Create variables for room and time slot assignments
        for course in self.courses:
            # Get valid rooms for this course - i.e. check valid capacity 
            valid_rooms = [i for i, room in enumerate(self.rooms) 
                        if room.capacity >= course.num_students]
            
            if not valid_rooms:
                raise ValueError(f"No suitable room found for {course.code}: {course.name} "
                            f"(needs room with capacity >= {course.num_students})")
            
            # Create room assignment variable
            self.room_assignments[course.code] = self.model.NewIntVarFromDomain(
                cp_model.Domain.FromValues(valid_rooms),
                f'room_for_{course.code}'
            )
            
            # Create time block assignment variables for each day
            self.time_block_assignments[course.code] = {}
            for day in DAYS:
                self.time_block_assignments[course.code][day] = self.model.NewIntVar(
                    0, len(TIME_BLOCKS) - 1,
                    f'time_block_for_{course.code}_{day}'
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
                        # Use day-specific time block variable
                        time_block_var = self.time_block_assignments[course.code][slot.day]
                        
                        # Create helper variable for the equality
                        time_block_matches = self.model.NewBoolVar(f'time_block_matches_{course.code}_{i}')
                        self.model.Add(time_block_var == block_idx).OnlyEnforceIf(time_block_matches)
                        self.model.Add(time_block_var != block_idx).OnlyEnforceIf(time_block_matches.Not())
                        
                        # NOT(slot_var) OR time_block_matches
                        self.model.AddBoolOr([slot_var.Not(), time_block_matches])
            
            # Constraint: Each course must have exactly num_sessions slots assigned - 3 seessions a week 
            self.model.Add(sum(self.slot_assignments[course.code].values()) == course.num_sessions)
            
            # Constraint: Each course must use exactly num_sessions different days - each class on a seperate day 
            self.model.Add(sum(self.day_assignments[course.code].values()) == course.num_sessions)
        
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
        """Add soft constraints for professor preferences"""
        # Add time exclusion as soft constraints with penalties
        for professor, excluded_time_days in self.professor_time_exclusions.items():
            # Find all courses taught by this professor
            prof_courses = [c for c in self.courses if c.professor == professor]
            
            if not prof_courses:
                continue
            
            weight = self.preference_weights.get(professor, 10)  # Default weight is 10
            
            # For each excluded time-day combination
            for hour, minute, excluded_days in excluded_time_days:
                # Find the time block index
                time_block_idx = None
                for idx, (block_hour, block_minute) in enumerate(TIME_BLOCKS):
                    if block_hour == hour and block_minute == minute:
                        time_block_idx = idx
                        break
                
                if time_block_idx is None:
                    continue  # Skip if time not found
                
                # Add penalty variables for each course on excluded days
                for course in prof_courses:
                    for day in excluded_days:
                        if day not in DAYS:
                            continue  # Skip invalid days
                        
                        # Create a penalty variable that will be 1 if preference is violated
                        penalty_var = self.model.NewBoolVar(f'penalty_{course.code}_{day}_{time_block_idx}')
                        
                        # If day is assigned AND time block equals excluded time, then penalty applies
                        day_var = self.day_assignments[course.code][day]
                        time_block_var = self.time_block_assignments[course.code][day]
                        
                        # Create helper variable for time block match
                        time_matches = self.model.NewBoolVar(f'time_matches_{course.code}_{day}_{time_block_idx}')
                        self.model.Add(time_block_var == time_block_idx).OnlyEnforceIf(time_matches)
                        self.model.Add(time_block_var != time_block_idx).OnlyEnforceIf(time_matches.Not())
                        
                        # If day is assigned AND time matches excluded time, then penalty is 1
                        # day_var AND time_matches => penalty_var
                        self.model.AddBoolAnd([day_var, time_matches]).OnlyEnforceIf(penalty_var)
                        
                        # If NOT(day_var AND time_matches), then penalty is 0
                        # NOT(day_var) OR NOT(time_matches) => NOT(penalty_var)
                        self.model.AddBoolOr([day_var.Not(), time_matches.Not()]).OnlyEnforceIf(penalty_var.Not())
                        
                        # Add to list of penalties with weight
                        self.preference_penalties.append((penalty_var, weight))
    
    
    def solve(self, time_limit_seconds: int = 60) -> Optional[Dict[str, ScheduleAssignment]]:
        """Solve the course scheduling problem with soft constraints"""
        start_time = time.time()
        
        # Create objective function from penalties
        if self.preference_penalties:
            objective_terms = []
            for penalty_var, weight in self.preference_penalties:
                objective_terms.append(penalty_var * weight)
            self.model.Minimize(sum(objective_terms))
        
        # Set time limit
        self.solver.parameters.max_time_in_seconds = time_limit_seconds
        
        # Solve
        status = self.solver.Solve(self.model, self.solution_printer)
        solve_time = time.time() - start_time

        # Determine if the solver stopped due to time limit
        # (The CP-SAT solver doesn't provide a direct flag, so we use the wall time.)
        self.timed_out = (self.solver.WallTime() >= time_limit_seconds)
        
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            if self.timed_out:
                print(f"Solution found in {solve_time:.2f} seconds, but the time limit was reached during search.")
            else:
                print(f"Solution found in {solve_time:.2f} seconds within the time limit.")
            
            # Calculate and display penalties if applicable
            if self.preference_penalties:
                total_penalty = 0
                violated_prefs = 0
                for penalty_var, weight in self.preference_penalties:
                    if self.solver.Value(penalty_var) == 1:
                        total_penalty += weight
                        violated_prefs += 1
                print(f"Total preference penalty: {total_penalty}")
                print(f"Preferences violated: {violated_prefs} out of {len(self.preference_penalties)}")
            
            return self._extract_solution()
        
        elif status == cp_model.INFEASIBLE:
            print("The problem is infeasible - no solution exists that satisfies all constraints")
            return None
        
        else:
            if self.timed_out:
                print("Solver stopped due to the time limit being reached, and no solution was found.")
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
                print(f"  num_students: {course.num_students}")
                print(f"  Days: {', '.join(assigned_days)}")
                
                # Display time for each day separately
                print(f"  Times: ", end="")
                for day in assigned_days:
                    block_idx = self.Value(self.time_block_assignments[course.code][day])
                    time_block = TIME_BLOCKS[block_idx]
                    print(f"{day} {time_block[0]:02d}:{time_block[1]:02d}", end=", " if day != assigned_days[-1] else "")
                print()
                
                print(f"  Schedule: {', '.join(str(ts) for ts in time_slots)}")
                print(f"  Room: {room.name} (Capacity: {room.capacity})")
                print()


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
            print(f"  num_students: {course.num_students}")
            print(f"  Schedule: {', '.join(str(ts) for ts in assignment.time_slots)}")
            print(f"  Room: {assignment.room.name} (Capacity: {assignment.room.capacity})")
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
    # Define rooms without enforcing room types
    rooms = [
        Room("A101", 30),
        Room("A102", 50),
        Room("B201", 100),
        Room("B202", 150),
        Room("C301", 200),
        Room("C302", 180),
        Room("D401", 35),
        Room("D402", 40),
        Room("D403", 30),
        Room("E501", 40),
        Room("E502", 60),
    ]
    
    # Define courses without room type requirements
    courses = [
        # PHYS courses
        Course("PHYS101", "Intro to Physics", 45, "Dr. Smith"),
        Course("PHYS102", "Mechanics", 30, "Dr. Johnson"),
        Course("PHYS110", "Physics Lab", 20, "Dr. Brown"),
        
        # MATH courses
        Course("MATH101", "Calculus I", 50, "Dr. Davis"),
        Course("MATH102", "Linear Algebra", 40, "Dr. Taylor"),
        
        # COMP courses
        Course("COMP101", "Intro to Programming", 60, "Dr. Anderson"),
        Course("COMP110", "Programming Lab", 25, "Dr. Thomas"),
    ]
    
    # Create and solve the scheduling problem
    scheduler = CourseScheduler(courses, rooms)
    
   # Add professor time exclusions with day specificity and weights
    scheduler.add_professor_time_exclusion("Dr. Smith", [
        (8, 30, ["Mon", "Tue", "Wed", "Thu", "Fri"]),  # No 8:30 AM classes any day
        (16, 30, ["Mon", "Wed"])  # No 4:30 PM classes on Monday and Wednesday
    ], weight=20)  # Higher weight = more important preferenc, senior prof for example

    scheduler.add_professor_time_exclusion("Dr. Brown", [
        (8, 30, ["Mon", "Tue", "Wed", "Thu", "Fri"]),  # No 8:30 AM classes any day
        (2, 30, ["Mon", "Wed"])  # No 12:30 PM classes on Monday and Wednesday
    ], weight=10)  # Regular weight preference
    
    # Build the model
    scheduler.build_model()
    
    # Solve
    solution = scheduler.solve(time_limit_seconds=60)
    
    # Print final solution
    print_final_solution(courses, solution)
    
    # Print schedule for Monday
    print_day_schedule("Mon", courses, solution)


if __name__ == "__main__":
    run_example()