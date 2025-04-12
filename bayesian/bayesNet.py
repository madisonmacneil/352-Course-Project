import pandas as pd
import numpy as np
import torch
import pyro
import pyro.distributions as dist

# Load course database and professor quality data
courses_df = pd.read_csv("data/Final_DB.csv")
profs_df   = pd.read_csv("data/prof_qaulity_info.csv")

print(courses_df.columns)
print(profs_df.columns)

# If multiple instructors are listed, take the first name
courses_df['primary_instructor'] = courses_df['instructor'].str.split(',').str[0].str.strip()

# Merge course data with professor ratings on instructor name
merged_df = pd.merge(courses_df, profs_df, left_on='primary_instructor', right_on='name', how='left')

# Inspect the merged result for a few courses
merged_df[['course_code', 'primary_instructor', 'rating_val', 'diff_level']].head(10)

# Impute missing course quality and difficulty with column means
merged_df['rating_val'].fillna(merged_df['rating_val'].mean(), inplace=True)
merged_df['diff_level'].fillna(merged_df['diff_level'].mean(), inplace=True)

merged_df.rename(columns={'rating_val': 'course_quality', 'diff_level': 'course_difficulty'}, inplace=True)

np.random.seed(42)  # for reproducibility

train_records = []
# Sample 50 random courses from the merged data to simulate (or use all available courses with instructor data)
sample_courses = merged_df.dropna(subset=['course_quality', 'course_difficulty']).sample(n=50, random_state=42)

for _, course in sample_courses.iterrows():
    course_code = course['course_code']
    quality    = course['course_quality']   # professor rating
    difficulty = course['course_difficulty']  # professor difficulty
    for _ in range(5):  # simulate 5 students per course
        # Student features
        strength = np.random.normal(0, 1)  # overall student strength
        aptitude = strength + np.random.normal(0, 0.5)  # subject aptitude
        early_bird = np.random.binomial(1, 0.5)  # 1 = early bird, 0 = night owl
        friends = np.random.binomial(1, 0.3)
        class_time = np.random.binomial(1, 0.5)  # 1 = morning class, 0 = afternoon
        # Determine mismatch: 1 if student's preference doesn't match class time
        mismatch = abs(early_bird - class_time)
        # Participation probability via logistic model
        logit_p = -1 + 1.0*friends - 2.0*mismatch + 0.5*strength
        participation_prob = 1 / (1 + np.exp(-logit_p))
        participation = np.random.binomial(1, participation_prob)
        # Simulate final numeric score
        score = 70  # baseline
        score += 5 * strength
        score += 3 * aptitude
        score += 2 * (quality - 3)       # center quality around 3 (average)
        score += -4 * (difficulty - 3)   # center difficulty around 3
        score += 5 * participation
        score += 2 * friends
        # Clamp score to [0, 100]
        score = max(0, min(100, score))
        # Map score to letter grade
        if score >= 90:   grade = "A+"
        elif score >= 85: grade = "A"
        elif score >= 80: grade = "A-"
        elif score >= 77: grade = "B+"
        elif score >= 73: grade = "B"
        elif score >= 70: grade = "B-"
        elif score >= 67: grade = "C+"
        elif score >= 63: grade = "C"
        elif score >= 60: grade = "C-"
        elif score >= 50: grade = "D"
        else:             grade = "F"
        # Record the simulated data
        train_records.append({
            "course_code": course_code,
            "student_strength": strength,
            "subject_aptitude": aptitude,
            "early_bird": early_bird,
            "friends_in_class": friends,
            "class_time_morning": class_time,
            "participation": participation,
            "course_quality": quality,
            "course_difficulty": difficulty,
            "final_grade": grade
        })

train_df = pd.DataFrame(train_records)
print(train_df.head(5))

grade_order = ["F","D","C-","C","C+","B-","B","B+","A-","A","A+"]
grade_to_idx = {grade: idx for idx, grade in enumerate(grade_order)}
train_df["grade_code"] = train_df["final_grade"].map(grade_to_idx)
train_df[["final_grade","grade_code"]].head(10)


# Convert training data to torch tensors
feature_cols = ["student_strength", "subject_aptitude", "participation", "course_quality", "course_difficulty"]
X = torch.tensor(train_df[feature_cols].values, dtype=torch.float)
Y = torch.tensor(train_df["grade_code"].values, dtype=torch.long)

num_features = X.shape[1]        # 5 features
num_grades   = len(grade_order)  # 11 possible grade categories

def grade_model(features, grades=None):
    # Define weight matrix and bias for the linear model of Grade logits.
    # Shape: weight [num_grades, num_features], bias [num_grades]
    weight = pyro.param("weight", lambda: torch.zeros(num_grades, num_features))
    bias   = pyro.param("bias", lambda: torch.zeros(num_grades))
    # Batch plate for data points
    with pyro.plate("data", features.shape[0]):
        # Compute logits for each grade category for each data point
        logits = (weight @ features.T).T + bias  # result shape [batch_size, num_grades]
        # Sample the final grade from a Categorical distribution with these logits
        pyro.sample("obs", dist.Categorical(logits=logits), obs=grades)


def guide(features, grades=None):
    # No sampling needed since all uncertainty is in pyro.param (handled by SVI automatically)
    pass


from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam

# Setup SVI for training
optimizer = Adam({"lr": 0.005})
svi = SVI(model=grade_model, guide=guide, optim=optimizer, loss=Trace_ELBO())

# Training loop
num_iterations = 1000
for epoch in range(num_iterations):
    loss = svi.step(X, Y)  # one step of gradient descent on the ELBO
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: ELBO loss = {loss:.2f}")

learned_weight = pyro.param("weight").detach().numpy()
learned_bias   = pyro.param("bias").detach().numpy()
print("Learned weight shape:", learned_weight.shape)
print("Sample weights for features:", learned_weight[:,0])  # weights for student_strength across grade categories for example

import torch.nn.functional as F

# Define the scenario features
ex_strength       = 0.0  # average strength
ex_aptitude       = 0.0  # average aptitude in subject
ex_participation  = 0.0  # likely not participating (due to mismatch)
ex_quality        = 3.5  # course quality (instructor rating)
ex_difficulty     = 3.7  # course difficulty

# Make a feature tensor (1 x num_features)
ex_features = torch.tensor([[ex_strength, ex_aptitude, ex_participation, ex_quality, ex_difficulty]], dtype=torch.float)

# Get learned parameters
weight = pyro.param("weight")
bias   = pyro.param("bias")

# Compute logits for the example
logits = (weight @ ex_features.T).T + bias  # shape [1, num_grades]
probs = F.softmax(logits, dim=1)            # convert to probabilities
probs_array = probs.detach().numpy().flatten()

# Display the probability for each grade category
for grade, p in zip(grade_order, probs_array):
    print(f"{grade}: {p*100:.1f}%")


def predict_grade_distribution(student_strength, subject_aptitude, early_bird, friends_in_class, course_quality, course_difficulty):
    # Determine participation if not directly given, by using our logistic model for P (optional step)
    class_time = 1  # assuming morning class for example, adjust or take as parameter as needed
    mismatch = abs(early_bird - class_time)
    logit_p = -1 + 1.0*friends_in_class - 2.0*mismatch + 0.5*student_strength
    participation_prob = 1 / (1 + np.exp(-logit_p))
    participation = 1 if participation_prob > 0.5 else 0
    # Create features tensor
    feat = torch.tensor([[student_strength, subject_aptitude, participation, course_quality, course_difficulty]], dtype=torch.float)
    logits = (pyro.param("weight") @ feat.T).T + pyro.param("bias")
    probs = F.softmax(logits, dim=1).detach().numpy().flatten()
    return {grade: float(f"{p*100:.2f}") for grade, p in zip(grade_order, probs)}

# Example split (if we had enough real data)
from sklearn.model_selection import train_test_split
train_data, test_data = train_test_split(train_df, test_size=0.2, random_state=0)
# (Then convert to tensors and train model on train_data, finally evaluate on test_data)

# Compute accuracy on training data
pred_probs = F.softmax((pyro.param("weight") @ X.T).T + pyro.param("bias"), dim=1)
pred_labels = pred_probs.argmax(dim=1).numpy()
true_labels = Y.numpy()
accuracy = (pred_labels == true_labels).mean()
print(f"Training accuracy: {accuracy*100:.1f}%")
