# 🎓 Queen’s University Course Assistant

A multi-modal academic advising platform built with symbolic, probabilistic, and neural AI models. Designed for Queen’s University students and staff, this assistant solves three critical problems in course planning:

1. **Timetable Scheduling** via a CSP model
2. **Grade Prediction** using a Bayesian Network
3. **Natural Language Course Search** with a Seq2Seq deep learning model

> Created for the CISC 352 course project at Queen’s University by Simon Nair and Madison MacNeil.

---

## 🧠 Project Overview

Modern course selection tools often fall short—rigid interfaces, no personalized guidance, and limited automation. Our assistant provides:

- A **constraint-based scheduler** that generates feasible and preference-aware timetables
- A **Bayesian model** to predict your chances of success in a course
- A **natural language search engine** that converts English questions into SQL

---

## 🗂️ Repository Structure

- `csp/` — CSP scheduling system
- `bayesian/` — Bayesian grade prediction system
- `deep_learning/` — NLP to SQL query engine
- `data/` — Raw and processed datasets
- `sql_dbs/` — SQLite course database

---

## 🧩 Components

### 1. CSP Scheduler (Symbolic AI)

- Location: `csp/`
- Features:
  - Room capacity and time-slot conflict management
  - Professor time preferences (soft constraints)
  - GUI views: full schedule, per-day, hypothetical student schedule
- Uses Google OR-Tools (CP-SAT Solver)

#### To Run:

```bash
python FinalCSP.py
```

---

### 2. Bayesian Grade Predictor (Statistical AI)

- Location: `bayesian/`
- Features:
  - Latent variables: Subject Aptitude, Course Quality, Student Strength, Participation
  - 10,000 sample simulations per query using Pyro
  - Outputs full letter grade distribution and GPA estimate

#### To Run:

```bash
python bayesianNetwork.py
```

---

### 3. Natural Language to SQL Engine (Neural AI)

- Location: `deep_learning/`
- Features:
  - Converts queries like "Show 4th year Biology courses" into SQL
  - Built with BERT encoder and LSTM decoder
  - Custom tokenizer and vocabulary

#### To Run:

```bash
python inference.py --question "What biology courses are open to second-year students?"
```

---

## 📊 Data Sources

- Queen’s course listings and prerequisites
- RateMyProfessors data for professor quality
- Classroom sizes from `queensu.ca/classrooms`
- Regex and LLM-assisted parsing for deep learning queries

---

## ⚠️ Limitations

- CSP scaling limited to ~150 courses
- Bayesian model uses simulated priors
- NL2SQL is sensitive to vocabulary drift

---

## 🛠️ Technologies Used

- Python (PyTorch, Pyro, OR-Tools)
- BERT, LSTM (HuggingFace)
- SQLite3
- Pandas, Selenium, Regex
- Tkinter for GUI

---

## 👨‍💻 Authors

- Simon Nair — `21scn1@queensu.ca`
- Madison MacNeil — `20mkm17@queensu.ca`
