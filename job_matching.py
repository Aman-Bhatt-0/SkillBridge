import nltk
from nltk.corpus import stopwords


def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    words = nltk.word_tokenize(text.lower())
    return ' '.join([word for word in words if word.isalnum() or '-' in word and word not in stop_words])


def match_resumes(job_description, resumes):
    job_skills = set(preprocess_text(job_description).split())  # Convert job description to set of words

    matched_resumes = []
    for resume in resumes:
        # Use dictionary keys instead of numeric indexes
        resume_skills = set(preprocess_text(
            f"{resume['programming_languages']} {resume['ml_tools']} {resume['dev_tools']} {resume['courseworks']} {resume['soft_skills']}").split())

        common_skills = job_skills.intersection(resume_skills)  # Find common skills

        match_score = (len(common_skills) / len(job_skills)) * 100 if job_skills else 0

        matched_resumes.append({
            "name": resume['name'],
            "email": resume['email'],
            "match_score": round(match_score, 2)
        })

    return sorted(matched_resumes, key=lambda x: x["match_score"], reverse=True)
