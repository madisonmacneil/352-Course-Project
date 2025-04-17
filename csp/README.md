# 🧠 CSP-Based Course Scheduler

This system generates conflict-free course schedules using a constraint satisfaction approach, tailored for Queen’s University. It uses Google OR-Tools and enforces both hard and soft constraints for realistic scheduling.

---

## 🔧 Features

- Conflict-free scheduling for 150+ courses
- Room capacity enforcement
- No same-year, same-department course overlaps
- Professor time preferences modeled as soft constraints
- GUI with multiple views (weekly, daily, student simulation)

---

## 📁 Key Files

- `FinalCSP.py`: Main script for loading data, solving, and displaying the solution
- `csp.py`: Core CSP solver logic using OR-Tools
- `display.py`: Tkinter GUI interface
- `prof_preferences.py`: Random preference generator
- `explore.py`: Dataset cleaner (limits dept-year pairings)
- `rooms.csv` / `course_csp_limited.csv`: Clean input datasets

---

## ▶️ How to Run

```bash
python FinalCSP.py
```

> ⚠️ Solving with large datasets will use 100% CPU. Runtime may vary from 30 seconds to 25 minutes depending on sample size and constraints.
>
> The time limit is set to 1500 seconds by default. On average, solving a 150-course subset takes under 10 minutes on a modern CPU. You can adjust this number to trade off between performance and realism.

---

## 🖥️ GUI Options

- **Full Schedule**
- **Daily View**
- **Hypothetical Student Schedule** (3 core + 2 elective courses)
