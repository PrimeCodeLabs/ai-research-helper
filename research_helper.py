import os
import logging
from datetime import datetime
import re
import gradio as gr
from typing import List, Dict, Tuple, Optional
from fastapi import FastAPI
import uvicorn
import sqlite3
import json
import hashlib
import asyncio
from tavily import AsyncTavilyClient
from langchain_community.llms import HuggingFaceHub

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

MODEL_CONFIG = {
    "analytical": {
        "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "temperature": 0.3,
        "max_new_tokens": 2048,
        "top_p": 0.9,
        "top_k": 50,
        "repetition_penalty": 1.1
    }
}

class PersistentCacheManager:
    def __init__(self, ttl=3600):
        self.conn = sqlite3.connect('cache.db')
        self.conn.execute("PRAGMA journal_mode = WAL")
        self._init_db()
        self.ttl = ttl

    def _init_db(self):
        self.conn.execute('''CREATE TABLE IF NOT EXISTS cache
                            (key TEXT PRIMARY KEY,
                             data TEXT,
                             timestamp DATETIME)''')

    def get_key(self, query: str) -> str:
        normalized = re.sub(r'\s+', ' ', query).strip().lower()
        return hashlib.sha256(normalized.encode()).hexdigest()

    def get(self, query: str) -> Optional[Dict]:
        key = self.get_key(query)
        cursor = self.conn.execute("SELECT data, timestamp FROM cache WHERE key=?", (key,))
        row = cursor.fetchone()
        if row:
            data_str, timestamp = row
            if (datetime.now() - datetime.fromisoformat(timestamp)).seconds < self.ttl:
                return json.loads(data_str)
        return None

    def set(self, query: str, value: Dict):
        key = self.get_key(query)
        data_str = json.dumps(value)
        timestamp = datetime.now().isoformat()
        self.conn.execute("REPLACE INTO cache (key, data, timestamp) VALUES (?, ?, ?)",
                         (key, data_str, timestamp))
        self.conn.commit()

class HybridSearchClient:
    def __init__(self):
        self.tavily = AsyncTavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        self.cache = PersistentCacheManager()

    async def search(self, query: str) -> Dict:
        cached = self.cache.get(query)
        if cached:
            return cached

        try:
            response = await self.tavily.search(
                query,
                search_depth="advanced",
                max_results=15,
                include_answer=True,
                include_raw_content=True,
                include_images=True,
                max_age=5*365*24*60*60
            )
            
            processed = {
                "results": response.get("results", []),
                "answer": response.get("answer", ""),
                "raw_content": response.get("raw_content", "")
            }
            
            self.cache.set(query, processed)
            return processed
            
        except Exception as e:
            logging.error(f"Search error: {str(e)}")
            return {"error": str(e)}

class MultiModelManager:
    def __init__(self):
        self.llm = HuggingFaceHub(
            repo_id=MODEL_CONFIG["analytical"]["model"],
            model_kwargs={
                "temperature": MODEL_CONFIG["analytical"]["temperature"],
                "max_new_tokens": MODEL_CONFIG["analytical"]["max_new_tokens"],
                "top_p": MODEL_CONFIG["analytical"]["top_p"],
                "top_k": MODEL_CONFIG["analytical"]["top_k"],
                "repetition_penalty": MODEL_CONFIG["analytical"]["repetition_penalty"],
            },
            huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_TOKEN")
        )

    async def generate(self, prompt: str) -> str:
        try:
            response = await asyncio.to_thread(self.llm.invoke, prompt)
            cleaned = re.sub(r'<\/?s>|\[INST\]|<<\/?SYS>>|<\/?INST>', '', response)
            return cleaned.split("<|eot_id|>")[0].strip()
        except Exception as e:
            logging.error(f"Generation error: {str(e)}")
            raise

class ResearchAgent:
    def __init__(self):
        self.model = MultiModelManager()
        self.search = HybridSearchClient()
        self.answer_template = (
            "<s>[INST] Create a comprehensive report in markdown format based on these sources. "
            "Use natural section headers without numbering. Focus on key facts and quantitative data. "
            "Formatting requirements:\n"
            "- Use ## for section headers\n"
            "- Use bullet points for lists\n"
            "- Bold important terms\n"
            "- Never mention sources or formatting instructions\n\n"
            "Query: {query}\n\n"
            "Source Materials:\n{results}\n"
            "[/INST]"
        )

    async def research(self, query: str) -> Dict:
        try:
            search_results = await self.search.search(query)
            if "error" in search_results:
                return {"error": search_results["error"]}
            return await self._generate_answer(query, search_results)
            
        except Exception as e:
            logging.error(f"Research error: {str(e)}")
            return {"error": str(e)}

    async def _generate_answer(self, query: str, results: Dict) -> Dict:
        context = "\n".join([
            f"Source {i+1}: {res['title']}\n"
            f"Content: {res['content'][:300].strip()}{'...' if len(res['content']) > 300 else ''}\n"
            for i, res in enumerate(results["results"])
        ])

        prompt = self.answer_template.format(query=query, results=context)
        
        try:
            response = await self.model.generate(prompt)
            cleaned_response = re.sub(
                r'(?:\(Source \d+\)|\[.*?\]|\d+\.\s|\-+\s*|`{3,}|##\s*Formatting\s+requirements|'
                r'Never\s+mention\s+sources\s+or\s+formatting\s+instructions)',
                '',
                response
            ).strip()
            
            if not cleaned_response:
                raise ValueError("Empty response from model")
                
            return {
                "content": cleaned_response,
                "sources": results["results"]
            }
        except Exception as e:
            logging.error(f"Generation error: {str(e)}")
            return self._handle_fallback(query, results)
    
    def _handle_fallback(self, query: str, results: Dict) -> Dict:
        summary = "\n".join([f"- {res['title']}: {res['content'][:100]}" for res in results["results"]])
        return {
            "content": f"**Preliminary Findings**\n\n{summary}",
            "sources": results["results"]
        }

def create_gradio_interface():
    research_agent = ResearchAgent()

    async def process_query(query: str, history: List[Tuple[str, str]]):
        try:
            response = await research_agent.research(query)
            if "error" in response:
                return history + [(query, f"**Error**: {response['error']}")], []
            
            answer = response["content"]
            sources = [[s["title"], s["url"]] for s in response.get("sources", [])]
            return history + [(query, answer)], sources
            
        except Exception as e:
            logging.error(f"UI Query Error: {str(e)}")
            return history + [(query, f"**Error**: {str(e)}")], []
    
    def clear_input():
        return ""

    with gr.Blocks(theme=gr.themes.Soft(), css=".markdown {max-width: 800px; margin: auto}") as interface:
        gr.Markdown("# 🔍 AI Research Assistant")
        
        with gr.Tabs():
            with gr.Tab("Chat"):
                chatbot = gr.Chatbot(
                    height=600,
                    render_markdown=True,
                    bubble_full_width=False
                )
                query_box = gr.Textbox(
                    label="Research Query",
                    placeholder="Enter your question...",
                    lines=2,
                    max_lines=5
                )
                with gr.Row():
                    submit_btn = gr.Button("Search", variant="primary")
                    clear_btn = gr.Button("Clear History")

            with gr.Tab("References"):
                sources_display = gr.DataFrame(
                    headers=["Title", "URL"],
                    datatype=["str", "markdown"],
                    col_count=(2, "fixed"),
                    interactive=False
                )

        submit_btn.click(
            fn=process_query,
            inputs=[query_box, chatbot],
            outputs=[chatbot, sources_display]
        ).then(
            fn=clear_input,
            inputs=[],
            outputs=[query_box]
        )

        query_box.submit(
            fn=process_query,
            inputs=[query_box, chatbot],
            outputs=[chatbot, sources_display]
        ).then(
            fn=clear_input,
            inputs=[],
            outputs=[query_box]
        )

        clear_btn.click(
            fn=lambda: [],
            inputs=[],
            outputs=[chatbot]
        )

    return interface

app = FastAPI(title="Research Assistant")
gradio_app = create_gradio_interface()
app = gr.mount_gradio_app(app, gradio_app, path="/")

if __name__ == "__main__":
    config = uvicorn.Config(app, host="0.0.0.0", port=8000)
    server = uvicorn.Server(config)
    asyncio.run(server.serve())