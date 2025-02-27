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
    pattern = r"Prerequisites?:?\s*(.*?)(?=\s*(Corequisites|Corequisite|Exclusion|Exclusions|Recommended|$))"
    match = re.search(pattern, text, re.IGNORECASE)  # re.IGNORECASE makes it case insensitive
    if match:
        extracted = match.group(1).strip()
        return extracted
    return None

for elem in df['requirements']: 
    prereq = extract_prerequisites(elem)
    prereqs.append(prereq)

df['prerequisites'] = prereqs


def extract_corequisites(text):
    pattern = r"Corequisites?:?\s*(.*?)(?=\s*(Exclusion|Exclusio|Exclusions|Recommended|$))"
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

def extract_exclusions(text):
    pattern = r"Exclusions?:?\s*(.*?)(?=\s*(?:Prerequisite|Corequisite|Corequisites|Recommended|$))"
    match = re.search(pattern, text, re.IGNORECASE)  # re.IGNORECASE makes it case insensitive
    if match:
        extracted = match.group(1).strip()
        return extracted
    return None

exclusions = []
for elem in df['requirements']: 
    exclusion = extract_exclusions(elem)
    exclusions.append(exclusion)

df['exclusions'] = exclusions

def extract_recommended(text):
    pattern = r"Recommended:?\s*(.*?)(?=\s*(Exclusion|Exclusions|Corequisite|Corequisites|$))"
    match = re.search(pattern, text, re.IGNORECASE) 
    if match:
        return match.group(1).strip()
    return None

recommendations = []
for elem in df['requirements']: 
    recommended = extract_recommended(elem)
    recommendations.append(recommended)

df['recommended'] = recommendations

df.to_csv('ProcessedDB.csv')


