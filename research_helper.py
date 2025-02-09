import os
import logging
import time
import concurrent.futures
import backoff
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

# Disable Gradio analytics
os.environ["GRADIO_ANALYTICS_ENABLED"] = "false"

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

from duckduckgo_search import DDGS
from langchain_community.llms import HuggingFaceHub

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
            age = (datetime.now() - datetime.fromisoformat(timestamp)).seconds
            if age < self.ttl:
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
    def __init__(self, max_results=15, min_delay=0.2):
        self.cache = PersistentCacheManager()
        self.ddg_max_results = max_results
        self.min_delay = min_delay
        self.last_request_time = 0.0
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
    
    def _wait_for_rate_limit(self):
        current_time = time.time()
        elapsed = current_time - self.last_request_time
        if elapsed < self.min_delay:
            time.sleep(self.min_delay - elapsed)
        self.last_request_time = time.time()

    @backoff.on_exception(
        backoff.expo,
        Exception,
        max_tries=3,
        max_time=30
    )
    def _safe_ddg_search(self, query: str) -> List[Dict]:
        self._wait_for_rate_limit()
        results_list = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=self.ddg_max_results):
                results_list.append({
                    "title": r.get("title", ""),
                    "url": r.get("href", ""),
                    "content": r.get("body", r.get("snippet", "")) or ""
                })
        return results_list

    def _concurrent_search(self, query: str) -> List[Dict]:
        future = self.executor.submit(self._safe_ddg_search, query)
        return future.result()

    async def search(self, query: str) -> Dict:
        cached = self.cache.get(query)
        if cached:
            return cached

        try:
            results = await asyncio.to_thread(self._concurrent_search, query)
            processed = {"results": results, "answer": "", "raw_content": ""}
            self.cache.set(query, processed)
            return processed
        except Exception as e:
            logging.error(f"Search error: {str(e)}")
            return {"error": str(e)}

    def __del__(self):
        self.executor.shutdown(wait=False)

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
            "- **Bold** important terms\n"
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
        if "results" not in results:
            return {"error": "No 'results' in search response."}

        context = []
        for i, res in enumerate(results["results"]):
            snippet = res['content'][:300].strip()
            snippet += "..." if len(res['content']) > 300 else ""
            context.append(f"Source {i+1}: {res['title']}\nContent: {snippet}\n")
        combined_context = "\n".join(context)

        prompt = self.answer_template.format(query=query, results=combined_context)

        try:
            response = await self.model.generate(prompt)
            cleaned_response = re.sub(
                r'^.*?Source \d+:.*?Content:.*?\n[#]+\s',
                r'## ',
                response,
                flags=re.DOTALL
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
        summary = "\n".join([
            f"- {res['title']}: {res['content'][:100]}"
            for res in results.get("results", [])
        ])
        return {
            "content": f"**Preliminary Findings**\n\n{summary}",
            "sources": results.get("results", [])
        }

def create_gradio_interface():
    research_agent = ResearchAgent()

    custom_css = """
    body { background-color: #000 !important; }
    .terminal { 
        background: #001100 !important; 
        color: #00ff00 !important; 
        font-family: 'Courier New', monospace !important;
        padding: 24px !important;
        border: 2px solid #00ff00 !important;
        border-radius: 8px !important;
        box-shadow: 0 0 20px rgba(0, 255, 0, 0.2) !important;
    }
    .chatbot {
        height: 500px !important;
        background: transparent !important;
        border: 1px solid #00ff0077 !important;
        overflow-y: auto !important;
    }
    .katex, .katex * {
        color: #00ff00 !important;
    }
    @media (max-width: 600px) {
        .chatbot { height: 400px !important; }
        .terminal { padding: 12px !important; }
    }
    """

    def clean_decorative_lines(content: str) -> str:
        def is_decorative(line: str) -> bool:
            stripped = line.strip()
            if len(stripped) >= 3 and all(c == stripped[0] for c in stripped):
                return stripped[0] in {'-', '=', '*', '_', '~', '#'}
            return False

        lines = content.split('\n')
        start = 0
        while start < len(lines) and is_decorative(lines[start]):
            start += 1
        end = len(lines) - 1
        while end >= 0 and is_decorative(lines[end]):
            end -= 1
        return '\n'.join(lines[start:end+1]) if start <= end else ""

    async def process_query(query: str, history: List[List[str]]) -> Tuple[List[List[str]], List[List[str]], str]:
        try:
            response = await research_agent.research(query)
            if "error" in response:
                new_history = history + [[query, f"⚠️ **ERROR**: {response['error']}"]]
                return new_history, [], ""
            
            cleaned_content = clean_decorative_lines(response['content'])
            answer = f"\n{cleaned_content}\n"
            sources = [[s["title"], s.get("url", "N/A")] for s in response.get("sources", [])]

            new_history = history + [[query, answer]]
            return new_history, sources, ""
        except Exception as e:
            err_msg = f"⚠️ **ERROR**: {str(e)}"
            new_history = history + [[query, err_msg]]
            return new_history, [], ""

    async def clear_interface():
        return [], [], ""

    with gr.Blocks(theme=gr.themes.Default(primary_hue="green"), css=custom_css) as interface:
        with gr.Column(elem_classes="terminal"):
            gr.Markdown("# 🔍 AI RESEARCH TERMINAL")
            
            with gr.Tabs():
                with gr.Tab("MAIN CONSOLE"):
                    chatbot = gr.Chatbot(
                        height=500,
                        bubble_full_width=False,
                        show_label=False,
                        elem_classes="chatbot",
                        render_markdown=True,
                        latex_delimiters=[
                            {"left": "$$", "right": "$$", "display": True},
                            {"left": "$", "right": "$", "display": False}
                        ]
                    )
                    
                    with gr.Row():
                        query_box = gr.Textbox(
                        show_label=False,
                        placeholder="Type your query here...",
                        container=False,
                        autofocus=True
                        )
                        with gr.Column(scale=1):
                            submit_btn = gr.Button("⚡ EXECUTE", variant="primary")
                            clear_btn = gr.Button("🔄 CLEAR", variant="secondary")
                
                with gr.Tab("DATA SOURCES"):
                    sources_display = gr.DataFrame(
                        headers=["DOCUMENT TITLE", "REFERENCE URL"],
                        datatype=["str", "markdown"],
                        col_count=(2, "fixed"),
                        interactive=False
                    )

            with gr.Row(elem_classes="status-bar"):
                gr.Markdown("□ STATUS: OPERATIONAL")
                gr.Markdown("□ MODEL: MIXTRAL-8x7B")
                gr.Markdown("□ SEARCH: DUCKDUCKGO")

        query_box.submit(
            process_query,
            inputs=[query_box, chatbot],
            outputs=[chatbot, sources_display, query_box]
        )
        submit_btn.click(
            process_query,
            inputs=[query_box, chatbot],
            outputs=[chatbot, sources_display, query_box]
        )
        clear_btn.click(
            clear_interface,
            outputs=[chatbot, sources_display, query_box]
        )

    return interface

app = FastAPI(title="Research Assistant")
gradio_app = create_gradio_interface()
app = gr.mount_gradio_app(app, gradio_app, path="/")

if __name__ == "__main__":
    config = uvicorn.Config(app, host="0.0.0.0", port=8000)
    server = uvicorn.Server(config)
    asyncio.run(server.serve())