"""
DokuHub — O'zbekcha Dokumentatsiya Platformasi
FastAPI backend with Markdown rendering, JSON quizzes, and AI Q&A
"""

import json
import os
import re
import math
from pathlib import Path
from collections import Counter

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

import markdown
from markdown.extensions.codehilite import CodeHiliteExtension
from markdown.extensions.tables import TableExtension
from markdown.extensions.fenced_code import FencedCodeExtension
from markdown.extensions.toc import TocExtension

# ==================================================
# App Setup
# ==================================================
app = FastAPI(title="DokuHub", description="O'zbekcha dokumentatsiya platformasi")

BASE_DIR = Path(__file__).resolve().parent
CONTENT_DIR = BASE_DIR / "content"

# Mount static files
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

# Templates
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


# ==================================================
# Content Loading Helpers
# ==================================================
def get_all_docs():
    """Load all documentation topics from content/ directory"""
    docs = []
    if not CONTENT_DIR.exists():
        return docs

    for folder in sorted(CONTENT_DIR.iterdir()):
        if not folder.is_dir():
            continue

        meta_file = folder / "meta.json"
        if not meta_file.exists():
            continue

        try:
            with open(meta_file, "r", encoding="utf-8") as f:
                meta = json.load(f)

            docs.append({
                "slug": folder.name,
                "meta": meta,
                "has_quiz": (folder / "quiz.json").exists(),
                "has_content": (folder / "index.md").exists(),
            })
        except (json.JSONDecodeError, IOError):
            continue

    # Sort by order field
    docs.sort(key=lambda d: d["meta"].get("order", 999))
    return docs


def load_markdown(slug: str):
    """Load and render markdown content for a topic"""
    md_file = CONTENT_DIR / slug / "index.md"
    if not md_file.exists():
        return None, []

    with open(md_file, "r", encoding="utf-8") as f:
        raw = f.read()

    # Configure markdown extensions
    md = markdown.Markdown(extensions=[
        FencedCodeExtension(),
        CodeHiliteExtension(css_class="highlight", guess_lang=True, use_pygments=False),
        TableExtension(),
        TocExtension(permalink=False, toc_depth=3, slugify=lambda value, separator: re.sub(r'[^\w\s-]', '', value.strip().lower()).replace(' ', separator)),
    ])

    html_content = md.convert(raw)

    # Build TOC from the generated toc
    toc_items = []
    toc_html = md.toc
    # Parse TOC items from the toc HTML
    for match in re.finditer(r'href="#([^"]+)"[^>]*>([^<]+)', toc_html):
        toc_id = match.group(1)
        toc_text = match.group(2)
        # Determine level by class or nesting (simple heuristic)
        toc_items.append({"id": toc_id, "text": toc_text, "level": 2})

    return html_content, toc_items


def load_quiz(slug: str):
    """Load quiz data for a topic"""
    quiz_file = CONTENT_DIR / slug / "quiz.json"
    if not quiz_file.exists():
        return None

    with open(quiz_file, "r", encoding="utf-8") as f:
        return json.load(f)


def load_meta(slug: str):
    """Load metadata for a topic"""
    meta_file = CONTENT_DIR / slug / "meta.json"
    if not meta_file.exists():
        return None

    with open(meta_file, "r", encoding="utf-8") as f:
        return json.load(f)


# ==================================================
# AI Search Engine (TF-IDF based)
# ==================================================
class AISearch:
    """Simple TF-IDF based search engine for documentation content"""

    def __init__(self):
        self.documents = {}  # slug -> raw markdown text
        self.paragraphs = []  # list of (slug, paragraph_text)
        self._loaded = False

    def _load(self):
        """Load all content into memory"""
        if self._loaded:
            return

        if not CONTENT_DIR.exists():
            self._loaded = True
            return

        for folder in CONTENT_DIR.iterdir():
            if not folder.is_dir():
                continue

            md_file = folder / "index.md"
            if not md_file.exists():
                continue

            with open(md_file, "r", encoding="utf-8") as f:
                text = f.read()

            self.documents[folder.name] = text

            # Split into paragraphs
            paragraphs = re.split(r'\n\n+', text)
            for para in paragraphs:
                cleaned = re.sub(r'```[\s\S]*?```', '', para)  # Remove code blocks
                cleaned = re.sub(r'[#*`\[\]\(\)>|_\-]', ' ', cleaned)  # Remove markdown syntax
                cleaned = cleaned.strip()
                if len(cleaned) > 20:
                    self.paragraphs.append((folder.name, para.strip(), cleaned))

        self._loaded = True

    def _tokenize(self, text):
        """Simple tokenization"""
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        tokens = text.split()
        # Remove very short tokens
        return [t for t in tokens if len(t) > 1]

    def _tf(self, term, tokens):
        """Term frequency"""
        return tokens.count(term) / max(len(tokens), 1)

    def _idf(self, term, all_token_lists):
        """Inverse document frequency"""
        containing = sum(1 for tokens in all_token_lists if term in tokens)
        if containing == 0:
            return 0
        return math.log(len(all_token_lists) / containing)

    def search(self, query: str, top_k: int = 3):
        """Search for the most relevant paragraphs"""
        self._load()

        if not self.paragraphs:
            return None, None

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return None, None

        # Tokenize all paragraphs
        para_tokens = [self._tokenize(p[2]) for p in self.paragraphs]

        # Calculate TF-IDF scores
        scores = []
        for i, tokens in enumerate(para_tokens):
            score = 0
            for term in query_tokens:
                tf = self._tf(term, tokens)
                idf = self._idf(term, para_tokens)
                score += tf * idf

            # Bonus for exact phrase match
            para_lower = self.paragraphs[i][2].lower()
            query_lower = query.lower()
            if query_lower in para_lower:
                score += 2.0

            # Bonus for multiple query terms matching
            matched_terms = sum(1 for t in query_tokens if t in tokens)
            if matched_terms > 1:
                score += matched_terms * 0.3

            scores.append((score, i))

        # Sort by score
        scores.sort(reverse=True)

        # Get top results
        best = scores[:top_k]
        if best[0][0] < 0.01:
            return None, None

        # Build answer from top paragraphs
        top_result = self.paragraphs[best[0][1]]
        slug = top_result[0]

        # Combine top paragraphs from the same document
        answer_parts = []
        for score, idx in best:
            if score < 0.01:
                break
            para = self.paragraphs[idx]
            # Clean the paragraph for display
            text = para[1]
            text = re.sub(r'^#+\s+', '', text)  # Remove heading markers
            text = re.sub(r'```[\s\S]*?```', '[kod bloki]', text)  # Simplify code blocks
            text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)  # Remove links
            text = re.sub(r'[*_]{1,2}', '', text)  # Remove bold/italic
            text = text.strip()
            if text and text not in answer_parts:
                answer_parts.append(text)

        answer = "\n\n".join(answer_parts[:3])

        # If answer is too long, truncate
        if len(answer) > 800:
            answer = answer[:797] + "..."

        return answer, slug


ai_search = AISearch()


# ==================================================
# Routes — Pages
# ==================================================
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    docs = get_all_docs()
    total_questions = 0
    for doc in docs:
        quiz = load_quiz(doc["slug"])
        if quiz:
            total_questions += len(quiz.get("questions", []))

    return templates.TemplateResponse("home.html", {
        "request": request,
        "docs": docs,
        "total_questions": total_questions,
    })


@app.get("/docs/{slug}", response_class=HTMLResponse)
async def doc_page(request: Request, slug: str):
    meta = load_meta(slug)
    if not meta:
        raise HTTPException(status_code=404, detail="Dokumentatsiya topilmadi")

    content, toc = load_markdown(slug)
    if content is None:
        raise HTTPException(status_code=404, detail="Kontent topilmadi")

    # Get prev/next docs
    all_docs = get_all_docs()
    current_idx = next((i for i, d in enumerate(all_docs) if d["slug"] == slug), -1)

    prev_doc = None
    next_doc = None
    if current_idx > 0:
        prev_doc = {"slug": all_docs[current_idx - 1]["slug"], "title": all_docs[current_idx - 1]["meta"]["title"]}
    if current_idx < len(all_docs) - 1:
        next_doc = {"slug": all_docs[current_idx + 1]["slug"], "title": all_docs[current_idx + 1]["meta"]["title"]}

    return templates.TemplateResponse("doc.html", {
        "request": request,
        "meta": meta,
        "content": content,
        "toc": toc,
        "slug": slug,
        "has_quiz": (CONTENT_DIR / slug / "quiz.json").exists(),
        "prev_doc": prev_doc,
        "next_doc": next_doc,
    })


@app.get("/docs/{slug}/quiz", response_class=HTMLResponse)
async def quiz_page(request: Request, slug: str):
    meta = load_meta(slug)
    if not meta:
        raise HTTPException(status_code=404, detail="Dokumentatsiya topilmadi")

    quiz_data = load_quiz(slug)
    if not quiz_data:
        raise HTTPException(status_code=404, detail="Test topilmadi")

    return templates.TemplateResponse("quiz.html", {
        "request": request,
        "meta": meta,
        "quiz_data": quiz_data,
        "slug": slug,
    })


# ==================================================
# Routes — API
# ==================================================
class AIQuestion(BaseModel):
    question: str


@app.post("/api/ai-ask")
async def ai_ask(data: AIQuestion):
    question = data.question.strip()
    if not question:
        return JSONResponse({"answer": "Iltimos, savol yozing.", "source": None})

    answer, source = ai_search.search(question)

    if answer is None:
        return JSONResponse({
            "answer": "Kechirasiz, bu savol bo'yicha dokumentatsiyalarda ma'lumot topilmadi. Boshqa savol berib ko'ring yoki dokumentatsiyalarni o'qib chiqing!",
            "source": None
        })

    return JSONResponse({
        "answer": answer,
        "source": source
    })


@app.get("/api/search")
async def search_docs(q: str = ""):
    query = q.strip().lower()
    if len(query) < 2:
        return JSONResponse({"results": []})

    docs = get_all_docs()
    results = []

    for doc in docs:
        meta = doc["meta"]
        title = meta.get("title", "").lower()
        desc = meta.get("description", "").lower()
        tags = " ".join(meta.get("tags", [])).lower()

        # Check if query matches
        score = 0
        if query in title:
            score += 10
        if query in desc:
            score += 5
        if query in tags:
            score += 3

        # Check individual words
        for word in query.split():
            if word in title:
                score += 2
            if word in desc:
                score += 1
            if word in tags:
                score += 1

        if score > 0:
            results.append({
                "slug": doc["slug"],
                "title": meta.get("title", ""),
                "icon": meta.get("icon", "📄"),
                "snippet": meta.get("description", "")[:100],
                "score": score,
            })

    # Also search in markdown content
    for doc in docs:
        md_file = CONTENT_DIR / doc["slug"] / "index.md"
        if md_file.exists():
            with open(md_file, "r", encoding="utf-8") as f:
                content = f.read().lower()
            if query in content:
                # Check if already in results
                existing = next((r for r in results if r["slug"] == doc["slug"]), None)
                if existing:
                    existing["score"] += 2
                else:
                    # Find a snippet around the match
                    idx = content.find(query)
                    start = max(0, idx - 50)
                    end = min(len(content), idx + 100)
                    snippet = content[start:end].replace('\n', ' ').strip()
                    if start > 0:
                        snippet = "..." + snippet
                    if end < len(content):
                        snippet += "..."

                    results.append({
                        "slug": doc["slug"],
                        "title": doc["meta"].get("title", ""),
                        "icon": doc["meta"].get("icon", "📄"),
                        "snippet": snippet,
                        "score": 2,
                    })

    # Sort by score and limit
    results.sort(key=lambda r: r["score"], reverse=True)
    return JSONResponse({"results": results[:10]})


@app.post("/api/quiz/{slug}/check")
async def check_quiz(slug: str, answers: dict):
    quiz_data = load_quiz(slug)
    if not quiz_data:
        raise HTTPException(status_code=404, detail="Test topilmadi")

    questions = quiz_data.get("questions", [])
    correct = 0
    results = []

    for q in questions:
        qid = str(q["id"])
        user_answer = answers.get(qid)
        is_correct = user_answer == q["correct"]
        if is_correct:
            correct += 1
        results.append({
            "id": q["id"],
            "correct": is_correct,
            "correct_answer": q["correct"],
            "explanation": q.get("explanation", "")
        })

    total = len(questions)
    score = round((correct / max(total, 1)) * 100)
    passed = score >= quiz_data.get("passing_score", 60)

    return JSONResponse({
        "score": score,
        "correct": correct,
        "total": total,
        "passed": passed,
        "results": results
    })


# ==================================================
# Run
# ==================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
