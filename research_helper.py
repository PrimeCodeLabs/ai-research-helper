import os
import logging
from datetime import datetime
import re
import gradio as gr
from typing import List, Dict, Tuple, Optional
from langchain_huggingface import HuggingFaceEndpoint
from langchain.prompts import PromptTemplate
from fastapi import FastAPI
import uvicorn
import sqlite3
import json
import hashlib
import asyncio
from tavily import AsyncTavilyClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

MODEL_CONFIG = {
    "analytical": {
        "repo_id": "meta-llama/Meta-Llama-3-8B",
        "temperature": 0.3,
        "max_new_tokens": 4096
    }
}

class PersistentCacheManager:
    def __init__(self, ttl=3600):
        self.conn = sqlite3.connect('cache.db')
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
                query=query,
                search_depth="advanced",
                max_results=10,
                include_answer=True,
                include_raw_content=True
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
        self.model = HuggingFaceEndpoint(
            repo_id=MODEL_CONFIG["analytical"]["repo_id"],
            task="text-generation",
            temperature=MODEL_CONFIG["analytical"]["temperature"],
            max_new_tokens=MODEL_CONFIG["analytical"]["max_new_tokens"],
            do_sample=True,
            return_full_text=False,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.1,
            huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_TOKEN")
        )

class ResearchAgent:
    def __init__(self):
        self.model = MultiModelManager().model
        self.search = HybridSearchClient()
        self.answer_template = PromptTemplate.from_template(
            "Given the search results, answer the query.\n"
            "Query: {query}\n"
            "Search Results:\n{results}\n\n"
            "Provide a comprehensive answer with sources:"
        )

    async def research(self, query: str) -> Dict:
        try:
            search_results = await self.search.search(query)
            if "error" in search_results:
                return {"error": search_results["error"]}
                
            if search_results.get("answer"):
                return self._format_answer(search_results, query)
                
            return await self._generate_answer(query, search_results)
            
        except Exception as e:
            logging.error(f"Research error: {str(e)}")
            return {"error": str(e)}

    def _format_answer(self, results: Dict, query: str) -> Dict:
        return {
            "content": f"{results['answer']}\n\nSources: {self._format_sources(results['results'])}",
            "sources": results["results"]
        }

    async def _generate_answer(self, query: str, results: Dict) -> Dict:
        context = "\n".join([
            f"Source {i+1}: {res['title']}\nContent: {res['content'][:200]}..." 
            for i, res in enumerate(results["results"])
        ])
        
        prompt = self.answer_template.format(query=query, results=context)
        
        try:
            response = await asyncio.to_thread(self.model.generate, [{"query": query, "context": context}])
            content = response.generations[0][0].text.strip()
            if not content:
                raise ValueError("Empty response from model")
                
            return {
                "content": f"{content}\n\n**Sources:**\n{self._format_sources(results['results'])}",
                "sources": results["results"]
            }
        except Exception as e:
            logging.error(f"Generation error: {str(e)}")
            return self._handle_fallback(query, results)
    
    def _handle_fallback(self, query: str, results: Dict) -> Dict:
        summary = "\n".join([f"- {res['title']}: {res['content'][:100]}" for res in results["results"]])
        return {
            "content": f"I found these relevant results:\n{summary}",
            "sources": results["results"]
        }

    def _format_sources(self, sources: List[Dict]) -> str:
        return "\n".join([f"{i+1}. {s['url']}" for i, s in enumerate(sources[:3])])

def create_gradio_interface():
    research_agent = ResearchAgent()

    async def process_query(query: str, history: List[Tuple[str, str]]):
        try:
            response = await research_agent.research(query)
            if "error" in response:
                return [(query, f"Error: {response['error']}")], []
            
            answer = response["content"]
            sources = [s["url"] for s in response.get("sources", [])][:3]
            return history + [(query, answer)], sources
            
        except Exception as e:
            logging.error(f"UI Query Error: {str(e)}")
            return history + [(query, f"Error: {str(e)}")], []
    
    def clear_input():
        return ""  # Only clear the input box, not the chat history

    with gr.Blocks(theme=gr.themes.Soft()) as interface:
        gr.Markdown("# AI Research Assistant")
        
        with gr.Tabs():
            with gr.Tab("Chat"):
                chatbot = gr.Chatbot(height=500, render_markdown=True)
                query_box = gr.Textbox(
                    label="Research Query",
                    placeholder="Enter your question...",
                    lines=2
                )
                with gr.Row():
                    submit_btn = gr.Button("Search", variant="primary")
                    clear_btn = gr.Button("Clear")
            
            with gr.Tab("Sources"):
                sources_display = gr.JSON(label="Top Sources", value=[], every=1)

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
            fn=clear_input,
            inputs=[],
            outputs=[query_box]
        )

    return interface


app = FastAPI(title="Research Assistant")
gradio_app = create_gradio_interface()
app = gr.mount_gradio_app(app, gradio_app, path="/")

if __name__ == "__main__":
    config = uvicorn.Config(app, host="0.0.0.0", port=8000)
    server = uvicorn.Server(config)
    asyncio.run(server.serve())