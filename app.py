import gradio as gr
from transformers import pipeline
import PyPDF2
import io

# Load AI model for text analysis
nlp = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")

def extract_text_from_pdf(pdf_file):
    """Extract text from PDF"""
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def analyze_resume(pdf_file, job_description):
    """Analyze resume against job description"""
    
    # Extract text from PDF
    resume_text = extract_text_from_pdf(pdf_file)
    
    # Extract entities (names, organizations, etc.)
    entities = nlp(resume_text[:1000])  # Limit to first 1000 chars
    
    # Simple keyword matching for skills
    common_skills = ["python", "javascript", "react", "node", "aws", "docker", 
                     "kubernetes", "sql", "machine learning", "ai", "data", 
                     "agile", "git", "api", "cloud", "azure", "gcp"]
    
    found_skills = [skill for skill in common_skills if skill.lower() in resume_text.lower()]
    
    # Check match with job description
    if job_description:
        jd_lower = job_description.lower()
        matching_skills = [skill for skill in found_skills if skill in jd_lower]
        match_score = (len(matching_skills) / len(found_skills) * 100) if found_skills else 0
    else:
        matching_skills = found_skills
        match_score = 0
    
    # Format output
    output = f"""
## RESUME ANALYSIS

### Extracted Skills:
{', '.join(found_skills) if found_skills else 'No common skills detected'}

### Key Entities Found:
{', '.join([f"{e['word']} ({e['entity_group']})" for e in entities[:5]])}

### Match Score: {match_score:.1f}%

### Matching Skills:
{', '.join(matching_skills) if matching_skills else 'Upload job description for comparison'}

### Suggestions:
- Add more specific technical skills
- Quantify your achievements with numbers
- Include relevant certifications
- Tailor resume to job description keywords
"""
    
    return output

# Create interface
demo = gr.Interface(
    fn=analyze_resume,
    inputs=[
        gr.File(label="Upload Resume (PDF)", file_types=[".pdf"]),
        gr.Textbox(label="Job Description (Optional)", lines=5, placeholder="Paste job description here for match analysis...")
    ],
    outputs=gr.Markdown(label="Analysis Results"),
    title="🎯 AI Resume Analyzer",
    description="Upload your resume and optionally paste a job description. AI will analyze your skills and provide match scoring.",
    examples=[],
)

demo.launch(share=True)
