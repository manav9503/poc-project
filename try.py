"""
LangGraph - Starter FastAPI app

Run:
1. pip install fastapi uvicorn spacy networkx python-multipart
2. python -m spacy download en_core_web_sm
3. uvicorn app:app --reload
"""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
import uuid
import spacy
import networkx as nx

# Load spaCy NLP model
nlp = spacy.load("en_core_web_sm")

# FastAPI app
app = FastAPI(title="LangGraph API")

# In-memory stores
DOCS = {}           # doc_id -> {title, text}
GRAPH = nx.MultiDiGraph()

# Response models
class UploadResponse(BaseModel):
    doc_id: str
    title: str

class ExtractResult(BaseModel):
    entities: list
    relations: list

# --- Utilities ---

def simple_text_from_upload(file: UploadFile) -> str:
    """Extract plain text from uploaded file (txt only for now)."""
    content = file.file.read()
    try:
        return content.decode("utf-8")
    except Exception:
        return ""

def extract_entities_relations(text: str):
    """NER + simple co-mention relation extraction."""
    doc = nlp(text)
    entities = []
    for ent in doc.ents:
        entities.append({
            "text": ent.text,
            "label": ent.label_,
            "start_char": ent.start_char,
            "end_char": ent.end_char
        })

    relations = []
    for sent in doc.sents:
        ents_in_sent = [e for e in sent.ents]
        if len(ents_in_sent) >= 2:
            for i in range(len(ents_in_sent)-1):
                subj = ents_in_sent[i]
                obj = ents_in_sent[i+1]
                relations.append({
                    "subject": subj.text,
                    "subject_type": subj.label_,
                    "predicate": "co-mention",
                    "object": obj.text,
                    "object_type": obj.label_,
                    "sentence": sent.text
                })
    return entities, relations

def add_to_graph(entities, relations, doc_id):
    """Add extracted entities/relations to in-memory graph."""
    for e in entities:
        node_id = f"ent:{e['text']}"
        if not GRAPH.has_node(node_id):
            GRAPH.add_node(node_id, label=e["label"], text=e["text"])
        GRAPH.add_edge(node_id, f"doc:{doc_id}", relation="MENTIONED_IN")

    for r in relations:
        s_id = f"ent:{r['subject']}"
        o_id = f"ent:{r['object']}"
        if not GRAPH.has_node(s_id):
            GRAPH.add_node(s_id, label=r["subject_type"], text=r["subject"])
        if not GRAPH.has_node(o_id):
            GRAPH.add_node(o_id, label=r["object_type"], text=r["object"])
        GRAPH.add_edge(s_id, o_id, relation=r["predicate"],
                       provenance={"doc_id": doc_id, "sentence": r.get("sentence")})

# --- API Endpoints ---

@app.post("/upload", response_model=UploadResponse)
async def upload(file: UploadFile = File(...), title: str = Form(...)):
    text = simple_text_from_upload(file)
    if not text:
        raise HTTPException(status_code=400, detail="Could not decode file as text")
    doc_id = str(uuid.uuid4())
    DOCS[doc_id] = {"title": title or file.filename, "text": text}
    return {"doc_id": doc_id, "title": title}

@app.post("/extract/{doc_id}", response_model=ExtractResult)
async def extract(doc_id: str):
    if doc_id not in DOCS:
        raise HTTPException(status_code=404, detail="Document not found")
    text = DOCS[doc_id]["text"]
    entities, relations = extract_entities_relations(text)
    add_to_graph(entities, relations, doc_id)
    return {"entities": entities, "relations": relations}

@app.get("/graph/nodes")
async def graph_nodes():
    return list(GRAPH.nodes(data=True))

@app.get("/graph/edges")
async def graph_edges():
    return list(GRAPH.edges(data=True))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
