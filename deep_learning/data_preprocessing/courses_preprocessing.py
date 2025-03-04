import pandas as pd 
import re


def extract_prerequisites(text):
    pattern = r"Prerequisites?:?\s*(.*?)(?=\s*(Corequisites|Corequisite|Exclusion|Exclusions|Recommended|$))"
    match = re.search(pattern, text, re.IGNORECASE)  # re.IGNORECASE makes it case insensitive
    if match:
        extracted = match.group(1).strip()
        return extracted
    return None


def extract_corequisites(text):
    pattern = r"Corequisites?:?\s*(.*?)(?=\s*(Exclusion|Exclusio|Exclusions|Recommended|$))"
    match = re.search(pattern, text, re.IGNORECASE)  # re.IGNORECASE makes it case insensitive
    if match:
        extracted = match.group(1).strip()
        return extracted
    return None


def extract_exclusions(text):
    pattern = r"Exclusions?:?\s*(.*?)(?=\s*(?:Prerequisite|Corequisite|Corequisites|Recommended|$))"
    match = re.search(pattern, text, re.IGNORECASE)  # re.IGNORECASE makes it case insensitive
    if match:
        extracted = match.group(1).strip()
        return extracted
    return None

def extract_recommended(text):
    pattern = r"Recommended:?\s*(.*?)(?=\s*(Exclusion|Exclusions|Corequisite|Corequisites|$))"
    match = re.search(pattern, text, re.IGNORECASE) 
    if match:
        return match.group(1).strip()
    return None

def remove_faculty_prefixes(text): 
    pattern = r".*?Offering Faculty:(Faculty of [^:]+)?\s*(.*)"
    match = re.search(pattern, text, re.IGNORECASE) 
    if match:
        return match.group(1).strip()
    return None

def remove_outcome_prefix(text): 
    pattern = r".*?Course Learning Outcomes:[^:]+)?\s*(.*)"
    match = re.search(pattern, text, re.IGNORECASE) 
    if match:
        return match.group(1).strip()
    return None



def main():

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
    coreqs = []
    exclusions = []
    recommendations = []
    cleaned_faculties = [] 
    cleaned_outcomes = []

    for elem in df['requirements']: 
        prereq = extract_prerequisites(elem)
        prereqs.append(prereq)

    df['prerequisites'] = prereqs

    for elem in df['requirements']: 
        coreq = extract_corequisites(elem)
        coreqs.append(coreq)

    df['corequisites'] = coreqs

    for elem in df['requirements']: 
        exclusion = extract_exclusions(elem)
        exclusions.append(exclusion)

    df['exclusions'] = exclusions

    for elem in df['requirements']: 
        recommended = extract_recommended(elem)
        recommendations.append(recommended)

    df['recommended'] = recommendations

    for elem in df['faculty']: 
        c_faculty = remove_faculty_prefixes(elem)
        cleaned_faculties.append(c_faculty)
    
    df['faculty'] = cleaned_faculties


    for elem in df['outcomes']: 
        outcome = remove_outcome_prefix(elem)
        cleaned_outcomes.append(outcome)
    
    df['outcomes'] = cleaned_outcomes

    df.to_csv('ProcessedDB.csv')


