# resume_api.py

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import tempfile
from script_resume import extract_text_from_pdf, score_resume_with_llm
app = FastAPI(title="Resume Evaluation API", description="Upload a resume PDF to evaluate and extract info.", version="1.0")


@app.post("/evaluate-resume")
async def evaluate_resume(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        return JSONResponse(status_code=400, content={"error": "Only PDF files are supported."})

    # Save to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        contents = await file.read()
        temp_file.write(contents)
        temp_path = temp_file.name

    # Process
    try:
        resume_text = extract_text_from_pdf(temp_path)
        result = score_resume_with_llm(resume_text)
        return result
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
