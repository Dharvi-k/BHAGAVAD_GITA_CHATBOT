from fastapi import FastAPI       # create web API endpoints
from pydantic import BaseModel    # define and validate request JSON structure
import uvicorn                    # run the FastAPI server
from chatbot import smart_query, extract_paragraph


# FastAPI app
app = FastAPI(title="Bhagavad Gita QA API", version="1.0")

# Request body schema
class QueryRequest(BaseModel):
    query: str

# Root endpoint
@app.get("/")
def home():
    return {"message": "Welcome to the Bhagavad Gita QA API 🙏"}

# Main query endpoint
@app.post("/ask")
def ask_gita(req: QueryRequest):
    answer = smart_query(req.query)
    return answer

# GET version (for browser testing)
@app.get("/ask")
def ask_gita_get(query: str):
  full_response = smart_query(query)
  answer=extract_paragraph(full_response)
  return answer

