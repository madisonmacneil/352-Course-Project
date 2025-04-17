# Bayesian Grade Prediction System 🧠📊

---

## 🔍 Overview

The system uses probabilistic inference to estimate the likelihood of a student receiving each possible letter grade (A+ to F) based on a rich set of inputs including:

- Major-course alignment
- Historical success with a given professor
- Professor difficulty and ratings
- Prerequisite grades
- Overall GPA
- Course load
- Class schedule preferences
- Social participation (e.g., friends in class)

Each grade is generated via sampling from a CPT (Conditional Probability Table) and then expanded into a 13-point letter scale using adaptive logic.

---

## 📁 Files

| File                                 | Description                                                                                                |
| ------------------------------------ | ---------------------------------------------------------------------------------------------------------- |
| `bayesianNetwork.py`               | Main implementation. Handles data loading, network definition, conditional sampling, and grade prediction. |
| `complete_courses.csv`             | Maps course codes to prerequisites and departments.                                                        |
| `category_map.txt`                 | Converts majors and departments into broad academic categories for compatibility modeling.                 |
| `major_course_success_matrix.json` | Contains historical success distributions (A-F) for major-course combinations.                             |
| `prof_qaulity_info.csv`            | Ratings and difficulty levels for professors, scraped from RateMyProfessors.                               |
| `course_csp.csv`                   | (Optional) Used to match courses to professors for more personalized predictions.                          |

---

## 🚀 How to Run

To use the Bayesian network:

Install dependencies:

```bash
pip install pyro-ppl torch pandas numpy
```


To execute the Bayesian grade prediction system:

Ensure all required data files are in place across the bayesian/, data/, and csp/ directories.

Run the script:

python bayesianNetwork.py

Follow the prompts to enter:

Course code (e.g., CISC352)

Your major, GPA, and course load

Your class schedule and whether you have friends in the course

Prerequisite grades for relevant courses

For filling in the prompts:

Course Code: Any course in course_csp.csv

Major: Engineering, Computer Science, Life/Health Sciences, Physical Sciences / Math, Social Sciences, Arts / Humanities, Languages / Literature, Business / Interdisciplinary, Fine Arts / Media

The system will return:

A full probability distribution of expected letter grades (A+ through F)

An estimated GPA outcome

An interpretation of predicted performance (e.g., "Above average performance expected")

Optionally, you can run a sensitivity analysis and compare test student profiles to evaluate the model’s fairness and flexibility. This was used to help adjust the weights. This was used to help with the tuning.
