#!/usr/bin/env python3
"""
Smart Interview Simulator — single-file app 🎤🤖  (v2: PDF reports + offline mode)
================================================================================
AI asks interview questions → you answer by VOICE → AI transcribes,
evaluates content + confidence → gives feedback → download a PDF report.

Skills demonstrated: Python · NLP · Speech Recognition · APIs (REST) · clean architecture

TWO MODES
---------
1) CLOUD  (default, best accuracy):
       pip install fastapi uvicorn openai python-multipart reportlab
       export OPENAI_API_KEY="sk-..."
       python smart_interview.py

2) OFFLINE (no API key, runs locally):
       pip install fastapi uvicorn python-multipart reportlab faster-whisper requests
       # install Ollama (https://ollama.com) then:  ollama pull llama3.1
       export INTERVIEW_MODE=offline
       python smart_interview.py

Then open http://localhost:8000
"""
from __future__ import annotations

import json
import os
import re
import sys
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from datetime import datetime
from io import BytesIO

try:
    import uvicorn
    from fastapi import FastAPI, UploadFile, File, Form
    from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
except ImportError:
    sys.exit("Run: pip install fastapi uvicorn python-multipart reportlab")

# --------------------------------------------------------------------------- #
#  0. Config — choose CLOUD or OFFLINE
# --------------------------------------------------------------------------- #
MODE = os.environ.get("INTERVIEW_MODE", "cloud").lower()   # "cloud" | "offline"


# --------------------------------------------------------------------------- #
#  1. LLM clients (swappable interface — clean architecture)
# --------------------------------------------------------------------------- #
class LLMClient(ABC):
    @abstractmethod
    def complete_json(self, system: str, user: str) -> dict: ...


class OpenAIClient(LLMClient):
    """CLOUD mode — GPT-4o via OpenAI API."""
    def __init__(self, model: str = "gpt-4o"):
        from openai import OpenAI
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            sys.exit("Set OPENAI_API_KEY, or use offline mode (INTERVIEW_MODE=offline).")
        self._client = OpenAI(api_key=key)
        self.model = model

    def complete_json(self, system: str, user: str) -> dict:
        resp = self._client.chat.completions.create(
            model=self.model, temperature=0.4,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
        )
        return json.loads(resp.choices[0].message.content)


class OllamaClient(LLMClient):
    """OFFLINE mode — local LLM via Ollama."""
    def __init__(self, model: str = "llama3.1", host: str = "http://localhost:11434"):
        self.model, self.host = os.environ.get("OLLAMA_MODEL", model), host

    def complete_json(self, system: str, user: str) -> dict:
        import requests
        r = requests.post(
            f"{self.host}/api/chat",
            json={
                "model": self.model, "stream": False, "format": "json",
                "options": {"temperature": 0.4},
                "messages": [{"role": "system", "content": system},
                             {"role": "user", "content": user}],
            }, timeout=180,
        )
        r.raise_for_status()
        return json.loads(r.json()["message"]["content"])


# --------------------------------------------------------------------------- #
#  2. Speech Recognition — cloud (Whisper API) or offline (faster-whisper)
# --------------------------------------------------------------------------- #
class Transcriber(ABC):
    @abstractmethod
    def transcribe(self, audio_path: str) -> str: ...


class WhisperAPITranscriber(Transcriber):
    def __init__(self):
        from openai import OpenAI
        self._client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    def transcribe(self, audio_path: str) -> str:
        with open(audio_path, "rb") as f:
            return self._client.audio.transcriptions.create(
                model="whisper-1", file=f, response_format="text").strip()


class LocalWhisperTranscriber(Transcriber):
    """Offline STT with faster-whisper (downloads a small model on first run)."""
    def __init__(self, size: str = "base"):
        from faster_whisper import WhisperModel
        self._model = WhisperModel(os.environ.get("WHISPER_SIZE", size),
                                   device="cpu", compute_type="int8")

    def transcribe(self, audio_path: str) -> str:
        segments, _ = self._model.transcribe(audio_path)
        return " ".join(s.text for s in segments).strip()


# --------------------------------------------------------------------------- #
#  3. NLP + acoustic confidence scoring (pure Python)
# --------------------------------------------------------------------------- #
FILLERS = ["um", "uh", "er", "ah", "like", "you know", "basically", "actually"]
HEDGES = ["maybe", "i think", "i guess", "sort of", "kind of", "probably", "perhaps"]


@dataclass
class ConfidenceReport:
    score: int
    filler_count: int
    hedge_count: int
    words_per_minute: float
    notes: list[str]


def score_confidence(transcript: str, duration_sec: float) -> ConfidenceReport:
    text = transcript.lower()
    words = re.findall(r"[a-z']+", text)
    n_words = len(words)
    fillers = sum(text.count(f) for f in FILLERS)
    hedges = sum(text.count(h) for h in HEDGES)
    wpm = (n_words / duration_sec * 60) if duration_sec > 0 else 0.0

    score, notes = 100, []
    score -= min(int(fillers / max(n_words, 1) * 300), 30)
    if fillers > 3:
        notes.append(f"Reduce filler words ({fillers} detected).")
    score -= min(hedges * 4, 20)
    if hedges > 2:
        notes.append("Sound more decisive — fewer hedging phrases.")
    if wpm and not (110 <= wpm <= 160):
        score -= 10
        notes.append(f"Pace was {wpm:.0f} wpm; aim for ~130 wpm.")
    if n_words < 15:
        score -= 10
        notes.append("Answer was quite short — add more detail/examples.")
    if not notes:
        notes.append("Clear, steady, and decisive delivery. Nice work!")

    return ConfidenceReport(max(0, min(100, score)), fillers, hedges,
                            round(wpm, 1), notes)


# --------------------------------------------------------------------------- #
#  4. AI Interviewer
# --------------------------------------------------------------------------- #
QUESTION_SYS = (
    "You are a senior technical interviewer. Generate ONE concise interview "
    'question as JSON: {"question": str, "skill": str}. '
    "Adapt to the role and do NOT repeat previous questions."
)
EVAL_SYS = (
    "You are an expert interview coach. Evaluate the candidate's answer. "
    'Return JSON: {"content_score": int, "relevance": str, '
    '"strengths": [str], "improvements": [str], "model_answer": str}.'
)


class Interviewer:
    def __init__(self, llm: LLMClient, role: str):
        self.llm, self.role, self.asked = llm, role, []

    def next_question(self) -> dict:
        prev = "; ".join(self.asked) or "none"
        data = self.llm.complete_json(
            QUESTION_SYS, f"Role: {self.role}. Previously asked: {prev}.")
        self.asked.append(data["question"])
        return data

    def evaluate(self, question, transcript, conf: ConfidenceReport) -> dict:
        content = self.llm.complete_json(
            EVAL_SYS, f"Role: {self.role}\nQuestion: {question}\nAnswer: {transcript}")
        overall = round(0.7 * content["content_score"] + 0.3 * conf.score)
        return {
            "overall_score": overall,
            "role": self.role,
            "question": question,
            "transcript": transcript,
            "content": content,
            "confidence": asdict(conf),
        }


# --------------------------------------------------------------------------- #
#  5. PDF report generation (reportlab)
# --------------------------------------------------------------------------- #
def build_pdf(result: dict) -> bytes:
    from reportlab.lib.pagesizes import LETTER
    from reportlab.lib.units import inch
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                    ListFlowable, ListItem)

    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=LETTER,
                            topMargin=0.7 * inch, bottomMargin=0.7 * inch)
    ss = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=ss["Title"], fontSize=20)
    h2 = ParagraphStyle("h2", parent=ss["Heading2"], textColor="#06c")
    body = ss["BodyText"]
    story = []

    c, cf = result["content"], result["confidence"]
    story += [
        Paragraph("Smart Interview — Feedback Report", h1),
        Paragraph(datetime.now().strftime("%B %d, %Y · %I:%M %p"), body),
        Spacer(1, 12),
        Paragraph(f"<b>Role:</b> {result['role']}", body),
        Paragraph(f"<b>Overall Score:</b> {result['overall_score']}/100", body),
        Spacer(1, 10),
        Paragraph("Question", h2),
        Paragraph(result["question"], body),
        Spacer(1, 8),
        Paragraph("Your Answer (transcribed)", h2),
        Paragraph(result["transcript"] or "<i>(no speech detected)</i>", body),
        Spacer(1, 8),
        Paragraph(f"Content — {c['content_score']}/100", h2),
    ]
    story.append(Paragraph("<b>Strengths</b>", body))
    story.append(ListFlowable([ListItem(Paragraph(s, body)) for s in c["strengths"]],
                              bulletType="bullet"))
    story.append(Paragraph("<b>Improvements</b>", body))
    story.append(ListFlowable([ListItem(Paragraph(s, body)) for s in c["improvements"]],
                              bulletType="bullet"))
    story += [
        Spacer(1, 8),
        Paragraph(f"Confidence — {cf['score']}/100", h2),
        Paragraph(f"Pace: {cf['words_per_minute']} wpm · "
                  f"Fillers: {cf['filler_count']} · Hedges: {cf['hedge_count']}", body),
        ListFlowable([ListItem(Paragraph(n, body)) for n in cf["notes"]],
                     bulletType="bullet"),
        Spacer(1, 8),
        Paragraph("Model Answer", h2),
        Paragraph(c["model_answer"], body),
    ]
    doc.build(story)
    buf.seek(0)
    return buf.read()


# --------------------------------------------------------------------------- #
#  6. FastAPI backend + embedded frontend
# --------------------------------------------------------------------------- #
app = FastAPI(title="Smart Interview Simulator")
_sessions: dict[str, Interviewer] = {}
_last_result: dict[str, dict] = {}   # session_id -> latest result (for PDF)


def _make_llm() -> LLMClient:
    return OllamaClient() if MODE == "offline" else OpenAIClient()


def _make_transcriber() -> Transcriber:
    return LocalWhisperTranscriber() if MODE == "offline" else WhisperAPITranscriber()


_transcriber: Transcriber | None = None


def _get_transcriber() -> Transcriber:
    global _transcriber
    if _transcriber is None:
        _transcriber = _make_transcriber()
    return _transcriber


def _session(sid: str, role: str) -> Interviewer:
    if sid not in _sessions:
        _sessions[sid] = Interviewer(_make_llm(), role)
    return _sessions[sid]


@app.get("/api/question")
def get_question(session_id: str = "default", role: str = "Software Engineer"):
    return _session(session_id, role).next_question()


@app.post("/api/answer")
async def submit_answer(
    session_id: str = Form("default"),
    role: str = Form("Software Engineer"),
    question: str = Form(...),
    duration_sec: float = Form(...),
    audio: UploadFile = File(...),
):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
        tmp.write(await audio.read())
        path = tmp.name
    try:
        transcript = _get_transcriber().transcribe(path)
        conf = score_confidence(transcript, duration_sec)
        result = _session(session_id, role).evaluate(question, transcript, conf)
        _last_result[session_id] = result
        return JSONResponse(result)
    finally:
        os.unlink(path)


@app.get("/api/report.pdf")
def get_report(session_id: str = "default"):
    result = _last_result.get(session_id)
    if not result:
        return JSONResponse({"error": "No evaluation yet."}, status_code=404)
    pdf = build_pdf(result)
    return StreamingResponse(
        BytesIO(pdf), media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=interview_report.pdf"})


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML_PAGE.replace("__MODE__", MODE.upper())


HTML_PAGE = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Smart Interview Simulator</title>
<style>
 body{font-family:system-ui;max-width:700px;margin:40px auto;padding:0 16px;color:#222}
 h1{margin-bottom:4px}
 button{padding:10px 18px;font-size:16px;border-radius:8px;border:0;cursor:pointer;margin-right:6px}
 button:disabled{opacity:.5;cursor:not-allowed}
 #rec{background:#e11;color:#fff}#next{background:#06c;color:#fff}#pdf{background:#093;color:#fff}
 select{padding:8px;font-size:15px;border-radius:8px}
 .card{background:#f5f5f7;border-radius:12px;padding:16px;margin-top:14px;line-height:1.5}
 .bar{height:10px;background:#ddd;border-radius:5px;overflow:hidden;margin-top:6px}
 .bar>i{display:block;height:100%;background:#06c}
 .muted{color:#666;font-size:14px}
 .tag{display:inline-block;background:#06c;color:#fff;border-radius:6px;padding:2px 8px;font-size:12px}
</style></head><body>
<h1>🎤 Smart Interview Simulator <span class="tag">__MODE__ MODE</span></h1>
<p class="muted">AI asks · you answer by voice · AI scores content + confidence · export PDF.</p>
<p>Role:
 <select id="role">
  <option>Software Engineer</option><option>Data Scientist</option>
  <option>Product Manager</option><option>Machine Learning Engineer</option>
  <option>Frontend Developer</option>
 </select>
 <button id="next">Get Question →</button>
</p>
<div class="card"><b>Question:</b> <span id="q">Choose a role and click "Get Question".</span></div>
<p style="margin-top:14px">
 <button id="rec" disabled>● Start Recording</button>
 <button id="pdf" disabled>⬇ Download PDF Report</button>
</p>
<div id="out"></div>
<script>
let mediaRec,chunks=[],startT=0,question="",recording=false;
const sid="s"+Math.random().toString(36).slice(2);
const $=id=>document.getElementById(id);

async function loadQ(){
 $('q').textContent="Loading…";$('out').innerHTML="";
 $('rec').disabled=true;$('pdf').disabled=true;
 const role=encodeURIComponent($('role').value);
 const d=await (await fetch(`/api/question?session_id=${sid}&role=${role}`)).json();
 question=d.question;
 $('q').textContent=`[${d.skill}] ${d.question}`;
 $('rec').disabled=false;
}
$('next').onclick=loadQ;

$('rec').onclick=async()=>{
 const btn=$('rec');
 if(!recording){
  let stream;
  try{stream=await navigator.mediaDevices.getUserMedia({audio:true});}
  catch(e){alert("Microphone access denied.");return;}
  mediaRec=new MediaRecorder(stream);chunks=[];startT=Date.now();
  mediaRec.ondataavailable=e=>chunks.push(e.data);
  mediaRec.onstop=submit;
  mediaRec.start();recording=true;btn.textContent="■ Stop & Evaluate";
 }else{
  mediaRec.stop();recording=false;btn.textContent="● Start Recording";
 }
};

$('pdf').onclick=()=>{window.open(`/api/report.pdf?session_id=${sid}`,'_blank');};

async function submit(){
 const dur=(Date.now()-startT)/1000;
 const blob=new Blob(chunks,{type:'audio/webm'});
 const fd=new FormData();
 fd.append('session_id',sid);fd.append('role',$('role').value);
 fd.append('question',question);fd.append('duration_sec',dur);
 fd.append('audio',blob,'a.webm');
 $('out').innerHTML="<div class='card'>Transcribing & evaluating…</div>";
 const d=await (await fetch('/api/answer',{method:'POST',body:fd})).json();
 const c=d.content,cf=d.confidence;
 $('out').innerHTML=`
  <div class="card"><b>Overall Score: ${d.overall_score}/100</b>
    <div class="bar"><i style="width:${d.overall_score}%"></i></div></div>
  <div class="card"><b>Your answer (transcribed):</b><br>${d.transcript}</div>
  <div class="card"><b>Content: ${c.content_score}/100</b>
    <p>✅ ${c.strengths.join('<br>✅ ')}</p>
    <p>🔧 ${c.improvements.join('<br>🔧 ')}</p></div>
  <div class="card"><b>Confidence: ${cf.score}/100</b>
    <p class="muted">Pace: ${cf.words_per_minute} wpm ·
       Fillers: ${cf.filler_count} · Hedges: ${cf.hedge_count}</p>
    <p>${cf.notes.join('<br>')}</p></div>
  <div class="card"><b>💡 Model answer:</b><br>${c.model_answer}</div>`;
 $('pdf').disabled=false;
}
</script></body></html>"""


# --------------------------------------------------------------------------- #
#  7. Entry point
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    print(f"Smart Interview Simulator [{MODE.upper()} mode] → http://localhost:8000")
    if MODE == "offline":
        print("Offline: using faster-whisper + Ollama (make sure Ollama is running).")
    uvicorn.run(app, host="0.0.0.0", port=8000)