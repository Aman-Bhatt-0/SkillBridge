import re
import os
import spacy
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from pdfminer.high_level import extract_text

# Download necessary NLTK resources
# Load NLP Model
nlp = spacy.load("en_core_web_sm")
stop_words = set(stopwords.words("english"))


# Function to Extract Text from PDF
def extract_text_from_pdf(pdf_path):
    return extract_text(pdf_path)


# Function to Extract Name using NER
def extract_name(text):
    lines = text.split("\n")

    for line in lines:
        clean_line = line.strip()
        # Ignore lines that contain "Email", "Phone", "Contact", etc.
        if clean_line and not re.search(r"\b(email|phone|contact|mobile|linkedin|github|leetcode)\b", clean_line, re.IGNORECASE):
            return clean_line  # Return first valid name line

    return "Name not found"

# Function to Extract Contact Details
def extract_contact_details(text):
    mobile_pattern = r"\b[6-9]\d{9}\b"
    email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"

    mobile_match = re.search(mobile_pattern, text)
    email_match = re.search(email_pattern, text)

    return {
        "Mobile Number": mobile_match.group() if mobile_match else "Mobile number not found",
        "Email": email_match.group() if email_match else "Email not found"
    }


# Function to Extract Social Links
def extract_social_links(text):
    pattern = r"(?:https?://)?(?:www\.)?([a-zA-Z0-9.-]+\.com/[^\s,]+)"
    matches = re.findall(pattern, text)

    return {match.split(".com")[0].capitalize(): f"https://{match}" for match in matches}


DEGREE_KEYWORDS = [
    "B.Tech", "BSc", "B.Sc", "B.E", "Bachelor of Technology", "Bachelor of Science",
    "M.Tech", "MSc", "M.Sc", "Master of Science", "Master of Technology",
    "MBA", "Master of Business Administration", "BBA", "Bachelor of Business Administration",
    "PhD", "Doctor of Philosophy", "MCA", "Master of Computer Applications", "BCA",
    "Diploma", "Associate Degree", "PG Diploma", "MS", "Master of Science"
]

import re

import re


def extract_degree(text):
    """Extracts degrees with specializations from resumes using regex and normalization."""

    degree_pattern = r"\b(B\.?\s?Tech|BSc|B\.?\s?Sc|B\.?\s?E|M\.?\s?Tech|MSc|M\.?\s?Sc|MBA|BBA|PhD|MCA|BCA)\b(?:\s+in\s+[A-Za-z\s]+)?"

    matches = re.findall(degree_pattern, text, re.IGNORECASE)

    found_degrees = set()  # Use a set to remove duplicates

    for match in matches:
        # Normalize: Remove extra spaces and standardize format
        clean_degree = match.replace(" ", "").replace("BTech", "B.Tech").replace("BSc", "B.Sc").replace("MTech",
                                                                                                        "M.Tech")

        found_degrees.add(clean_degree.lower())  # Convert to lowercase for consistency

    # Convert back to a list with title case formatting (e.g., 'b.tech' → 'B.Tech')
    return [deg.title() for deg in found_degrees] if found_degrees else ["Not Found"]


# Keyword categories
programming_languages = [
    # Popular General-Purpose Languages
    "c", "c++", "c#", "java", "python", "javascript", "typescript", "swift", "kotlin",
     "rust", "dart", "php", "ruby", "perl", "matlab", "objective-c", "scala",

    # Web Development Languages
    "html", "css","react", "jsx", "tsx", "php", "asp.net", "coldfusion",

    # Scripting Languages
    "bash", "powershell", "groovy",  "vbscript",

    # Database Query Languages
    "sql", "pl/sql", "mysql", "postgresql", "mongodb", "redis", "cassandra", "oracle",
    "sqlite", "mariadb", "couchdb", "neo4j", "hive", "bigquery", "clickhouse",

    # Machine Learning & Data Science
    "julia", "stata", "wolfram", "haskell", "prolog", "lisp", "erlang",

    # Functional Programming
    "elixir", "clojure", "scheme", "ocaml",

    # Systems & Embedded Programming
    "assembly", "verilog", "vhdl", "fortran", "cobol", "modula-2", "pascal",

    # Game Development
    "unityscript", "godot", "unreal script", "haxe",

    # Blockchain & Smart Contracts
    "solidity", "vyper", "move", "chaincode",

    # Configuration & Markup Languages (Not Traditional Programming)
    "yaml", "json", "xml"
]
ml_tools = [
    # ✅ Popular ML Frameworks
    "tensorflow", "keras", "pytorch", "scikit-learn", "theano", "mxnet", "cntk", "chainer",

    # ✅ Data Manipulation & Processing
    "pandas", "numpy", "dask", "modin", "vaex", "polars",

    # ✅ Data Visualization
    "matplotlib", "seaborn", "plotly", "bokeh", "ggplot", "altair", "holoviews",

    # ✅ Deep Learning
    "fastai", "huggingface transformers", "torchvision", "detectron2", "timm", "fairseq",

    # ✅ Natural Language Processing (NLP)
    "spacy", "nltk", "gensim", "textblob", "stanfordnlp", "flair", "allennlp", "word2vec",
    "sentence-transformers", "sumy", "textract", "polyglot",

    # ✅ Computer Vision (CV)
    "opencv", "dlib", "pillow", "albumentations", "mmdetection", "detectron2", "yolov5",
    "deepface", "mediapipe", "imgaug",

    # ✅ Reinforcement Learning
    "gym", "stable-baselines3", "ray rllib", "openai gym", "dopamine", "trfl", "coach",

    # ✅ AutoML
    "h2o.ai", "tpot", "autokeras", "auto-sklearn", "mlbox", "pycaret", "datarobot",

    # ✅ Time Series Forecasting
    "prophet", "statsmodels", "sktime", "gluonts", "darts", "orbit", "pmdarima",

    # ✅ Feature Engineering
    "featuretools", "tsfresh", "boruta", "mlxtend", "scikit-feature", "feature-engine",

    # ✅ Model Explainability & Fairness
    "shap", "lime", "eli5", "fairlearn", "aif360", "dalex", "captum",

    # ✅ Hyperparameter Optimization
    "optuna", "hyperopt", "ray tune", "scikit-optimize", "spearmint", "bayesian-optimization",

    # ✅ Edge AI & Deployment
    "onnx", "tf-lite", "mlflow", "tensorRT", "torchscript", "tfx", "bentoML",

    # ✅ Graph Machine Learning
    "networkx", "pyg (pytorch geometric)", "dgl (deep graph library)", "graph-tool",

    # ✅ Cloud-Based ML Tools
    "aws sagemaker", "google ai platform", "azure ml studio", "vertex ai",

    # ✅ Anomaly Detection
    "pyod", "isolation forest", "hdbscan", "lof (local outlier factor)", "elliptic envelope",

    # ✅ Quantum Machine Learning
    "pennylane", "qiskit", "cirq", "tensorflow quantum",

    # ✅ Other Emerging ML Libraries
    "lightgbm", "xgboost", "catboost", "snap ml", "ngboost"
]
dev_tools = [
    # ✅ Version Control
    "git", "github", "gitlab", "bitbucket", "svn", "mercurial",

    # ✅ CI/CD & DevOps
    "docker", "kubernetes", "jenkins", "ansible", "terraform", "circleci", "travisci",
    "teamcity", "argoCD", "bamboo", "droneCI",

    # ✅ Code Editors & IDEs
    "vscode", "pycharm", "intellij idea", "eclipse", "atom", "sublime text",
    "vim", "notepad++", "jupyter notebook", "rider", "clion", "brackets",

    # ✅ Cloud Platforms
    "aws", "azure", "google cloud platform", "digitalocean", "oracle cloud",
    "ibm cloud", "heroku", "firebase", "netlify", "vercel",

    # ✅ Web Development Frameworks
    "node.js", "django", "flask", "fastapi", "express.js", "spring boot",
    "ruby on rails", "laravel", "asp.net", "vue.js", "react.js", "angular",

    # ✅ Containerization & Virtualization
    "vagrant", "lxc", "openvz", "podman", "helm",

    # ✅ Monitoring & Logging
    "prometheus", "grafana", "datadog", "splunk", "new relic", "elk stack",
    "graylog", "jaeger", "zipkin",

    # ✅ Database Management
    "mysql", "postgresql", "mongodb", "redis", "sqlite", "dynamodb",
    "cassandra", "neo4j", "cockroachdb", "orientdb",

    # ✅ Security & Testing
    "sonarqube", "metasploit", "owasp zap", "kali linux", "burp suite",
    "selenium", "pytest", "junit", "cypress", "postman",

    # ✅ API Development & Documentation
    "swagger", "postman", "graphql", "rest api", "grpc",

    # ✅ Data Science & Analytics
    "hadoop", "spark", "hive", "kafka", "airflow", "power bi", "tableau",
    "looker", "metabase"
]
courseworks = [
            "data structures and algorithms", "object-oriented programming", "artificial intelligence", "machine learning", "deep learning", "data science", "database management systems", "operating systems", "computer networks", "software engineering", "cloud computing", "cybersecurity", "web development", "mobile app development", "blockchain", "internet of things", "embedded systems", "devops",
    # ✅ Core Computer Science
    "data structures", "algorithms", "operating systems", "computer networks",
    "database management systems", "compiler design", "distributed systems",
    "object-oriented programming", "software engineering",

    # ✅ Machine Learning & AI
    "machine learning", "deep learning", "artificial intelligence", "reinforcement learning",
    "computer vision", "natural language processing", "big data analytics",
    "explainable AI", "AI ethics",

    # ✅ Web & App Development
    "full stack development", "frontend development", "backend development",
    "mobile app development", "progressive web apps", "react development",
    "flutter development", "android development", "ios development",

    # ✅ Cloud Computing & DevOps
    "cloud computing", "aws cloud practitioner", "google cloud fundamentals",
    "devops", "site reliability engineering", "cloud security",
    "serverless computing", "kubernetes administration",

    # ✅ Cybersecurity
    "ethical hacking", "penetration testing", "network security",
    "web application security", "blockchain security", "cryptography",

    # ✅ Embedded Systems & IoT
    "internet of things", "embedded systems", "robotics",
    "arduino programming", "raspberry pi development",

    # ✅ Blockchain & Cryptography
    "blockchain development", "cryptography", "smart contracts",
    "decentralized finance (DeFi)", "nft development",

    # ✅ Business & Soft Skills
    "entrepreneurship", "project management", "agile methodologies",
    "business analytics", "design thinking"
]
soft_skills = [
    "communication", "active listening", "public speaking", "negotiation", "presentation skills",
    "teamwork", "collaboration", "leadership", "mentoring", "coaching", "critical thinking",
    "problem-solving", "decision making", "strategic thinking", "emotional intelligence",
    "adaptability", "time management", "work ethic", "self-motivation", "conflict resolution",
    "stress management", "resilience", "creativity", "innovation", "multitasking",
    "attention to detail", "goal setting", "prioritization", "resource management",
    "growth mindset", "continuous learning", "self-discipline", "optimism", "accountability",
    "networking", "diplomacy", "storytelling", "self-confidence", "constructive feedback",
    "persuasion", "initiative", "self-awareness", "cultural intelligence", "decision-making under pressure",
    "project management", "risk assessment", "customer service", "ethics", "diversity awareness",
    "relationship-building", "handling criticism", "de-escalation", "creativity in problem-solving",
    "critical reading", "knowledge application", "remote collaboration", "virtual communication",
    "online etiquette", "cybersecurity awareness", "resourcefulness", "mediation",
    "empathy", "curiosity", "ability to learn quickly", "troubleshooting"
]


all_keywords = programming_languages + ml_tools + dev_tools + courseworks + soft_skills


# Function to Extract Keywords with Stopword Removal

def extract_keywords(text):
    text_lower = text.lower()

    matched_keywords = [kw.capitalize() for kw in all_keywords if kw in text_lower]

    return {
        "Programming Languages": [kw for kw in matched_keywords if kw.lower() in programming_languages],
        "Machine Learning Tools": [kw for kw in matched_keywords if kw.lower() in ml_tools],
        "Development Tools": [kw for kw in matched_keywords if kw.lower() in dev_tools],
        "Courseworks": [kw for kw in matched_keywords if kw.lower() in courseworks],
        "Soft Skills": [kw for kw in matched_keywords if kw.lower() in soft_skills]
    }

# Function to Parse Resume
def parse_resume(pdf_path):
    resume_text = extract_text_from_pdf(pdf_path)
    return {
        "name": extract_name(resume_text),
        "contact_details": extract_contact_details(resume_text),
        "social_links": extract_social_links(resume_text),
        "degree": extract_degree(resume_text),
        "keywords": extract_keywords(resume_text),
    }

