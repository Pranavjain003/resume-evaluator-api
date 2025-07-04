import gradio as gr
import google.generativeai as genai
import pdfplumber
import os
import json
from dotenv import load_dotenv
load_dotenv()

# Setup Gemini Flash Model
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise EnvironmentError("Please set GOOGLE_API_KEY environment variable.")
genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-1.5-flash")

# Helper: Extract text from PDF
def extract_text_from_pdf(pdf_file):
    with pdfplumber.open(pdf_file) as pdf:
        return "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])

# Core Logic to Score Resume
def score_resume_with_llm(resume_text):
    prompt = f"""
You are an intelligent resume evaluator. Given the resume text below, extract key candidate details and assign a **quality score between 0.0 and 1.0**. Be precise and strict in evaluation — do not award score if evidence is weak or missing.

---

📄 Return the following as valid JSON:

- name
- education
- experience_summary
- skills (list)
- projects_summary
- github_or_portfolio_links (list) — extract all valid URLs, especially GitHub, portfolio, LinkedIn
- certifications (list)
- participation (societies, hackathons, etc.)
- score (float between 0.0 and 1.0)
- tier: "high" (score ≥ 0.75), "medium" (0.4–0.74), "low" (≤ 0.39)
- tags (list): choose from ["focused", "project_ready", "poor_formatting", "incomplete", "well_presented", "github_present", "certified", "inactive_profile", "diverse_skills", "academic"]

---

📊 Scoring Criteria (Max 20 raw points → normalized to 0.0–1.0):

🔍 **Profile Depth & Quality (10 pts)**
- 2–3 focused domains (e.g., ML, web, systems) → +4  
- 6+ unrelated areas → −2  
- Skills backed by real projects → +5  
- Skills with no project evidence → 0  
- Clean formatting and structure → +3  
- Poor formatting or messy resume → −3  

💼 **Technical Strength (6 pts)**
- GitHub or portfolio with real projects → +5  
- Missing or empty links → 0  
- Certifications (Coursera, Google, etc.) → +2  
- Participation in hackathons/societies → +2  

🎓 **Academic Background (4 pts)**
- Tier 1 college → +4  
- Tier 2 → +2  
- Tier 3/unknown → 0  

---

📌 Instructions:
- Normalize total score: `normalized_score = round(raw_score / 20, 2)`
- Strictly extract GitHub/portfolio URLs — do NOT guess.
- If info is missing, leave field empty or null.
- Respond with valid JSON only.

---

📝 Resume Content:
\"\"\"{resume_text}\"\"\"
"""


    try:
        response = model.generate_content(prompt)
        result = response.text.strip()

        print("\n🧠 RAW LLM OUTPUT:\n", result)

        # Strip formatting from LLM output
        if result.startswith("```json"):
            result = result[len("```json"):].strip()
        if result.endswith("```"):
            result = result[:-3].strip()

        result = result.strip("`").strip()
        return json.loads(result)

    except Exception as e:
        print("❌ Error parsing JSON:", e)
        return None

# Gradio Interface Function
def process_resume(pdf_file):
    if not pdf_file:
        return "❌ Please upload a PDF resume.", None

    resume_text = extract_text_from_pdf(pdf_file)
    result = score_resume_with_llm(resume_text)

    if not result:
        return "❌ Could not score resume.", None

    with open("scored_resumes.json", "a", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
        # f.write(",\n")

    return (
        "✅ Resume scored successfully!",
        json.dumps(result, indent=2)
    )

# Gradio App
if __name__ == "__main__":
    gr.Interface(
        fn=process_resume,
        inputs=gr.File(label="Upload Resume (PDF Only)"),
        outputs=[
            gr.Textbox(label="Status"),
            gr.Textbox(label="Scored Resume JSON")
        ],
        title="📄 Resume Evaluator & Scorer",
        description="Upload a resume PDF. Gemini 1.5 Flash extracts info, scores it out of 20, and returns JSON output with tags and tier.",
    ).launch(server_port=7865)

