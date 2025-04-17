# 📊 Bayesian Grade Prediction

A probabilistic model that predicts grade distributions in Queen’s University courses using student background, class structure, and professor data.

---

## 🔍 What It Does

- Uses a Bayesian Network built with Pyro (PyTorch)
- Models latent variables: Subject Aptitude, Course Quality, Student Strength, Participation
- Computes a distribution over final grades (A+ to F)
- Estimates expected GPA

---

## 📁 Key Files

- `bayesianNetwork.py`: Core grade prediction script
- `category_map.txt`: Maps majors and departments to categories
- `major_course_success_matrix.json`: Success probabilities per major-course combo
- `complete_courses.csv`: Prerequisite and department info
- `prof_qaulity_info.csv`: Ratings and difficulty per professor

---

## ▶️ How to Run

```bash
python bayesianNetwork.py
```

The script will prompt for:

- Course code
- Major
- GPA
- Course load
- Prerequisite grades
- Morning/evening preference
- Social context (friends in class)

For filling in the prompts:

Course Code: Any course in course_csp.csv

Major: Engineering, Computer Science, Life/Health Sciences, Physical Sciences / Math, Social Sciences, Arts / Humanities, Languages / Literature, Business / Interdisciplinary, Fine Arts / Media

The system will return:

A full probability distribution of expected letter grades (A+ through F)

An estimated GPA outcome

An interpretation of predicted performance (e.g., "Above average performance expected")

Optionally, you can run a sensitivity analysis and compare test student profiles to evaluate the model’s fairness and flexibility. This was used to help adjust the weights. This was used to help with the tuning.
