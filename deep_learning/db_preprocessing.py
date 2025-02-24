import pandas as pd 
import re

df = pd.read_csv('courses.csv')
print(df.head(5))

#df has 2911 observations 
print(f"Size before removing duplicates: {df.shape}")
df = df.drop_duplicates()
print(f"Size after removing duplicates: {df.shape}")

#df columns
print(df.columns)

#make units numerical 
df['units'] = df['units'].str.strip('Units: ') 
pd.to_numeric(df['units'])

df['requirements'] = df['requirements'].str.strip('Requirements: ')
df['requirements'] = df['requirements'].astype(str)  

prereqs = []

def extract_prerequisites(text):
    pattern = r"Prerequisites?:?\s*(.*?)(?=\s*(Corequisites|Exclusions|$))"
    match = re.search(pattern, text, re.IGNORECASE)  # re.IGNORECASE makes it case insensitive
    if match:
        extracted = match.group(1).strip()
        return extracted
    return None

for elem in df['requirements']: 
    prereq = extract_prerequisites(elem)
    prereqs.append(prereq)

df['prerequisites'] = prereqs

print(df.head(5))

def extract_corequisites(text):
    pattern = r"Corequisites?:?\s*(.*?)(?=\s*(Exclusions|$))"
    match = re.search(pattern, text, re.IGNORECASE)  # re.IGNORECASE makes it case insensitive
    if match:
        extracted = match.group(1).strip()
        return extracted
    return None

coreqs = []
for elem in df['requirements']: 
    coreq = extract_corequisites(elem)
    coreqs.append(coreq)

df['corequisites'] = coreqs

print(df.head(5))


