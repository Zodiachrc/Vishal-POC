from flask import Flask, request, render_template, jsonify
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain
import os
import uuid
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'  
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GEMINI_API_KEY)

SESSIONS = {}  

FIRST_QUESTION_TEMPLATE = """Thank you for uploading your resume. Based on your listed experiences, projects, and technical skills, please answer the following questions as part of your AI Engineer interview:
(Only ask one question, do not summarize the resume. Assume you're the interviewer assessing the candidate's suitability for an AI Engineer role.)

Resume:

{resume_text}"""

NEXT_QUESTION_TEMPLATE = """You're an AI hiring manager. Continue the interview based on the following conversation so far. Ask ONE new and relevant technical or behavioral question for the AI Engineer position. Only ask the next question, nothing else.

Resume:
{resume_text}

Interview So Far:
{chat_history}

Ask only the next question."""

ASSESSMENT_TEMPLATE = """You're an AI interviewer. The following are a candidate's responses to questions in an AI Engineer interview:

{chat_history}

Here is the resume:
{resume_text}

Based on the responses, provide a short evaluation of the candidate's strengths, areas to improve, and their suitability for the AI Engineer role.
Respond in a professional, concise tone with 3 bullet points."""

first_question_chain = LLMChain(llm=llm, prompt=PromptTemplate(template=FIRST_QUESTION_TEMPLATE, input_variables=["resume_text"]))
next_question_chain = LLMChain(llm=llm, prompt=PromptTemplate(template=NEXT_QUESTION_TEMPLATE, input_variables=["resume_text", "chat_history"]))
assessment_chain = LLMChain(llm=llm, prompt=PromptTemplate(template=ASSESSMENT_TEMPLATE, input_variables=["chat_history", "resume_text"]))

@app.route('/')
def index():
    return render_template('index.html', state="start")

@app.route('/upload', methods=['POST'])
def upload_resume():
    if 'resume' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['resume']
    if file.filename == '':
        return jsonify({"error": "Empty file selected"}), 400
    
    filename = f"{uuid.uuid4().hex}.pdf"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)
    
    try:
        loader = PyPDFLoader(filepath)
        pages = loader.load()
        resume_text = " ".join([page.page_content for page in pages])
        
        first_question = first_question_chain.run(resume_text=resume_text)
        
        session_id = str(uuid.uuid4())
        SESSIONS[session_id] = {
            "resume_text": resume_text,
            "chat_history": f"Q1: {first_question}\n",
            "question_num": 1,
            "filepath": filepath
        }
        
        return render_template(
            'index.html', 
            state="question",
            question=first_question,
            question_num=1,
            session_id=session_id
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/answer', methods=['POST'])
def process_answer():
    answer = request.form.get('answer')
    session_id = request.form.get('session_id')
    
    if session_id not in SESSIONS:
        return jsonify({"error": "Session expired"}), 400
    
    session = SESSIONS[session_id]
    current_q = session["question_num"]
    
    session["chat_history"] += f"A{current_q}: {answer}\n"  
    
    if current_q < 5:
        next_q = next_question_chain.run(  
            resume_text=session["resume_text"],
            chat_history=session["chat_history"]
        )
        
        current_q += 1
        session["question_num"] = current_q
        session["chat_history"] += f"Q{current_q}: {next_q}\n"
        
        return render_template(
            'index.html',
            state="question",
            question=next_q,
            question_num=current_q,
            session_id=session_id
        )
    else:
        assessment = assessment_chain.run(
            chat_history=session["chat_history"],
            resume_text=session["resume_text"]
        )
        
        if os.path.exists(session["filepath"]):
            try:
                os.remove(session["filepath"])
            except:
                pass
        
        SESSIONS.pop(session_id, None)
        
        return render_template(
            'index.html',
            state="assessment",
            assessment=assessment
        )

if __name__ == '__main__':
    app.run(debug=True)

# Code comments:
# This Flask application creates an AI-powered interview system
# - Uses Gemini API to generate interview questions based on uploaded resume
# - Stores API key in environment variables using python-dotenv
# - Creates unique session for each interview
# - Processes 5 question-answer rounds before generating final assessment
# - Manages file uploads and cleanup after interview completion
# - The UPLOAD_FOLDER stores temporary resume files
# - Each prompt template defines how the AI generates questions and assessments
