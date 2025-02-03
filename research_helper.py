import os
import logging
import re
from datetime import datetime
import gradio as gr
from langchain_community.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict
from fastapi import FastAPI
import uvicorn

# For an alternative search library (optional)
try:
    from googlesearch import search as google_search_function
except ImportError:
    google_search_function = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Ensure the Hugging Face API token is set
HF_TOKEN = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
if not HF_TOKEN:
    logging.error("HUGGINGFACEHUB_API_TOKEN environment variable not set!")
    raise ValueError("HUGGINGFACEHUB_API_TOKEN must be set")

# ------------------------------
# MODEL SETUP
# ------------------------------

# Text generation models for chat and structured outputs
REPO_ID = "meta-llama/Llama-2-7b-chat-hf"

hf_llm = HuggingFaceHub(
    repo_id=REPO_ID,
    task="text-generation",
    model_kwargs={
        "temperature": 0.3,
        "max_new_tokens": 768,
        "top_p": 0.95,
        "repetition_penalty": 1.15,
        "top_k": 50,
        "stream": False,
        "do_sample": True
    }
)

hf_llm_json = HuggingFaceHub(
    repo_id=REPO_ID,
    task="text-generation",
    model_kwargs={
        "temperature": 0.05,
        "max_new_tokens": 256,
        "top_p": 0.85,
        "repetition_penalty": 1.0,
        "top_k": 10,
        "stream": False,
        "do_sample": False
    }
)

# ------------------------------
# SEARCH SETUP
# ------------------------------

# DuckDuckGo search wrapper (API-based)
wrapper = DuckDuckGoSearchAPIWrapper(
    max_results=10,
    time="m",
    safesearch="off",
    region="wt-wt",
    backend="api"
)
duckduckgo_search = DuckDuckGoSearchRun(api_wrapper=wrapper)

# ------------------------------
# PROMPT TEMPLATES
# ------------------------------

router_prompt = PromptTemplate(
    template="""[INST] You are a routing expert. Decide if this question needs real-time web search for accurate information.
Rules:
- Return "web_search" for:
    * Recent/ongoing events
    * Current information
    * Dynamic data
    * Questions with time-based keywords (current, today, latest, recent, etc.)
- Return "generate" otherwise
Response must be exactly "web_search" or "generate".

Question: {question}

Response: [/INST]""",
    input_variables=["question"],
)

query_prompt = PromptTemplate(
    template="""[INST] Transform the following question into search terms that will find the most current and accurate information.
Rules:
- Extract only the essential and relevant search terms
- Exclude filler words, focus on key names, places, and concepts
- Ensure the search terms align with the latest information and research trends

Question: {question}

Search terms: [/INST]""",
    input_variables=["question"],
)

generate_prompt = PromptTemplate(
    template="""[INST] You are a research assistant. Answer the question based on the following context and your knowledge.
Rules:
- Use the provided context when available
- Structure your response clearly
- Include relevant details and facts, and always indicate when information is unverifiable or unclear
- State explicitly if the answer cannot be confirmed with the provided context

Question: {question}

Context: {context}

Answer: [/INST]""",
    input_variables=["question", "context"],
)

summarize_prompt = PromptTemplate(
    template="""[INST] You are an expert summarizer. Summarize the following context into a concise summary focusing on the key points relevant to the research query.

Context:
{context}

Summary: [/INST]""",
    input_variables=["context"],
)

# Create a summarization chain
summarize_chain = summarize_prompt | hf_llm | StrOutputParser()

# Two generation chains: one for "Chat" mode and one for "Structured" (JSON) responses
chat_generate_chain = generate_prompt | hf_llm | StrOutputParser()
structured_generate_chain = generate_prompt | hf_llm_json | StrOutputParser()

# ------------------------------
# CHAINS SETUP & STATE DEFINITION
# ------------------------------

class GraphState(TypedDict, total=False):
    question: str
    generation: str
    search_query: str
    context: str
    history: str         # Conversation history as a string
    model_choice: str    # "Chat" or "Structured"
    search_provider: str # "DuckDuckGo" or "Google"

def extract_links(text: str) -> str:
    urls = re.findall(r'https?://\S+', text)
    if urls:
        urls = [url.rstrip('.,!?"\'') for url in urls]
        return "\n\nSources:\n" + "\n".join(f"- {url}" for url in urls)
    return ""

# ------------------------------
# AGENT FUNCTIONS
# ------------------------------

def transform_query(state: GraphState) -> dict:
    try:
        question = state["question"]
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M")
        q_lower = question.lower()
        if any(word in q_lower for word in ["trending", "current", "latest", "now", "today"]):
            search_query = question.replace("?", "").strip() + f" {current_time} latest"
            for loc in ["uk", "usa", "europe", "asia", "global"]:
                if loc in q_lower:
                    search_query += f" {loc}"
        else:
            response = query_prompt | hf_llm_json | StrOutputParser()
            transformed = response.invoke({"question": question})
            search_query = transformed.split("\n")[0].replace("[INST]", "").replace("[/INST]", "").strip()
            if len(search_query) < 5:
                search_query = question.replace("?", "").strip()
            if any(term in q_lower for term in ["this year", "next year", "last year"]):
                current_year = datetime.now().year
                if "next year" in q_lower:
                    search_query += f" {current_year + 1}"
                elif "last year" in q_lower:
                    search_query += f" {current_year - 1}"
                else:
                    search_query += f" {current_year}"
        logging.info(f"Transformed query: {search_query}")
        return {"search_query": search_query}
    except Exception as e:
        logging.error(f"Query transform error: {str(e)}")
        return {"search_query": state["question"].replace("?", "").strip() + " latest"}

def generate_answer(state: GraphState) -> dict:
    try:
        question = state["question"]
        logging.info(f"Generating answer for: {question}")
        raw_context = state.get("context", "")
        conversation_history = state.get("history", "")
        if conversation_history:
            raw_context = f"Previous conversation:\n{conversation_history}\n\nCurrent context:\n{raw_context}"
        if "No good DuckDuckGo Search Result" in raw_context:
            return {"generation": "I couldn't retrieve current information. Please verify from official sources."}
        formatted_context = "\n".join(
            f"- {line.strip()}" for line in raw_context.split("\n") 
            if line.strip() and not line.lower().startswith(('http', 'www'))
        )
        links = extract_links(raw_context)
        q_lower = question.lower()
        if any(word in q_lower for word in ["trending", "current", "latest", "now", "today"]):
            formatted_context += "\nNote: Only include current/recent info."
        if formatted_context:
            formatted_context = f"{formatted_context}\n{links}"
        if state.get("model_choice", "Chat") == "Structured":
            generation = structured_generate_chain.invoke({"question": question, "context": formatted_context})
        else:
            generation = chat_generate_chain.invoke({"question": question, "context": formatted_context})
        cleaned_response = generation.replace("[INST]", "").replace("[/INST]", "")
        cleaned_response = re.sub(r'Question:.*?Context:', '', cleaned_response, flags=re.DOTALL)
        if "Answer:" in cleaned_response:
            answer = cleaned_response.split("Answer:")[-1].strip()
        else:
            answer = cleaned_response.strip()
        answer = re.sub(r"Given these points,?|Based on these points,?|From these points,?", "", answer)
        answer = re.sub(r"Based on the (above|following|provided|available) (points|information|data),?", "", answer)
        if any(word in q_lower for word in ["trending", "current", "latest", "now", "today"]):
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M")
            answer = f"\n{answer}"
        if answer:
            answer = answer[0].upper() + answer[1:]
        return {"generation": "Answer:\n\n" + answer}
    except Exception as e:
        logging.error(f"Generation error: {str(e)}")
        return {"generation": "I encountered an error. Please try rephrasing your question."}

def do_duckduckgo_search(state: GraphState) -> dict:
    max_attempts = 3
    query = state["search_query"]
    if query in SEARCH_CACHE:
        logging.info(f"Cache hit for query: {query}")
        return {"context": SEARCH_CACHE[query]}
    for attempt in range(max_attempts):
        try:
            logging.info(f"DuckDuckGo attempt {attempt+1} for '{query}'")
            if not query or len(query) < 3:
                raise ValueError("Invalid search query")
            results = duckduckgo_search.invoke(query)
            if results and len(results.strip()) >= 10:
                if len(results) > 1000:
                    logging.info("Summarizing long DuckDuckGo results...")
                    summary = summarize_chain.invoke({"context": results})
                    results = summary.strip()
                SEARCH_CACHE[query] = results
                return {"context": results}
            if attempt < max_attempts - 1:
                state["search_query"] = f"{query} latest research studies"
            else:
                return {"context": "Note: Couldn't retrieve sufficient information. Try checking the latest sources."}
        except Exception as e:
            logging.error(f"DuckDuckGo search error: {str(e)}")
            if attempt == max_attempts - 1:
                return {"context": "Note: Couldn't retrieve sufficient information. Please try again."}

def do_google_search(state: GraphState) -> dict:
    query = state["search_query"]
    results = ""
    try:
        if google_search_function:
            for url in google_search_function(query, num_results=5):
                results += url + "\n"
            if not results.strip():
                results = "No results found."
        else:
            results = "Google search not available (install googlesearch-python)."
        return {"context": results}
    except Exception as e:
        return {"context": f"Google search error: {str(e)}"}

def do_web_search(state: GraphState) -> dict:
    provider = state.get("search_provider", "DuckDuckGo")
    if provider == "Google":
        return do_google_search(state)
    else:
        return do_duckduckgo_search(state)

def route_question(state: GraphState) -> str:
    try:
        question = state["question"].lower()
        time_keywords = [
            "current", "today", "latest", "now", "recent", "this year", "next year",
            "this season", str(datetime.now().year), str(datetime.now().year + 1),
            "will", "going to", "plans to", "upcoming", "future"
        ]
        if any(keyword in question for keyword in time_keywords):
            return "websearch"
        raw_output = router_prompt | hf_llm_json | StrOutputParser()
        decision = raw_output.invoke({"question": state["question"]}).strip().lower()
        if "web_search" in decision or "web search" in decision:
            return "websearch"
        return "generate"
    except Exception as e:
        logging.error(f"Routing error: {str(e)}")
        return "websearch"

# ------------------------------
# WORKFLOW DEFINITION
# ------------------------------

SEARCH_CACHE = {}

workflow = StateGraph(GraphState)
workflow.add_node("websearch", do_web_search)
workflow.add_node("transform_query", transform_query)
workflow.add_node("generate", generate_answer)
workflow.set_conditional_entry_point(route_question, {"websearch": "transform_query", "generate": "generate"})
workflow.add_edge("transform_query", "websearch")
workflow.add_edge("websearch", "generate")
workflow.add_edge("generate", END)

local_agent = workflow.compile()

# ------------------------------
# AGENT ENTRY POINT (Chat UI)
# ------------------------------

def run_agent(query: str, history: list = [], model_choice: str = "Chat", search_provider: str = "DuckDuckGo"):
    try:
        history_str = "\n".join([f"User: {u}\nAssistant: {a}" for u, a in history])
        initial_state = {
            "question": query,
            "generation": "",
            "search_query": "",
            "context": "",
            "history": history_str,
            "model_choice": model_choice,
            "search_provider": search_provider
        }
        output_state = local_agent.invoke(initial_state)
        answer = output_state.get("generation", "")
        new_history = history + [[query, answer]]
        return answer, new_history, None
    except Exception as e:
        return f"Error: {str(e)}", history, "error"

# ------------------------------
# HELPER FUNCTIONS FOR UI CONTROLS
# ------------------------------

def process_query(query, history, model_choice, search_provider):
    if not query.strip():
        return history, history
    answer, new_history, _ = run_agent(query, history, model_choice, search_provider)
    return new_history, new_history

def clear_chat():
    return [], []

def get_conversation_text(history):
    return "\n\n".join([f"User: {u}\nAssistant: {a}" for u, a in history])

# ------------------------------
# UI SETUP WITH TABS
# ------------------------------

custom_css = """
:root {
    --font-mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
    --font-sans: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
}
body {
    background-color: rgb(17, 24, 39);
    color: rgb(229, 231, 235);
}
.container {
    max-width: 100%;
    margin: 0 auto;
    padding: 1rem;
    box-sizing: border-box;
    font-family: var(--font-sans);
}
.title {
    text-align: left;
    margin-bottom: 1.5rem;
    font-family: var(--font-sans);
    font-size: 1.875rem;
    font-weight: 600;
    color: rgb(229, 231, 235);
    padding-bottom: 0.5rem;
    border-bottom: 1px solid rgb(55, 65, 79);
}
.output-container {
    background-color: rgb(17, 24, 39);
    border-radius: 6px;
    padding: 1rem;
    margin-bottom: 1rem;
    font-family: var(--font-mono);
    border: 1px solid rgb(55, 65, 79);
    height: calc(100vh - 300px);
    min-height: 300px;
    max-height: 500px;
    overflow-y: auto;
    -webkit-overflow-scrolling: touch;
}
.chatbot {
    height: 100%;
}
.input-area {
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
    margin-top: 1rem;
    font-family: var(--font-mono);
}
.input-box {
    flex-grow: 1;
    font-family: var(--font-mono);
    font-size: 0.875rem;
    background-color: rgb(17, 24, 39);
    border: 1px solid rgb(55, 65, 79);
    border-radius: 6px;
    color: rgb(229, 231, 235);
    padding: 0.5rem;
}
.button-row {
    display: flex;
    gap: 0.5rem;
}
.submit-button, .clear-button, .copy-button {
    font-family: var(--font-sans);
    min-height: 2.5rem;
    border-radius: 6px;
    background-color: rgb(75, 85, 99);
    color: rgb(229, 231, 235);
    font-weight: 500;
    cursor: pointer;
    transition: background-color 0.2s;
}
.submit-button:hover, .clear-button:hover, .copy-button:hover {
    background-color: rgb(55, 65, 79);
}
.text-area {
    width: 100%;
    background-color: rgb(17, 24, 39);
    border: 1px solid rgb(55, 65, 79);
    border-radius: 6px;
    color: rgb(229, 231, 235);
    padding: 0.5rem;
    font-family: var(--font-mono);
    font-size: 0.875rem;
    min-height: 100px;
}
@media (min-width: 768px) {
    .container {
        max-width: 800px;
        padding: 2rem;
    }
    .output-container {
        height: calc(100vh - 350px);
        max-height: 600px;
    }
    .input-area {
        flex-direction: row;
        align-items: flex-start;
    }
    .input-box {
        margin-right: 0.75rem;
    }
    .submit-button, .clear-button, .copy-button {
        width: auto;
        padding: 0 2rem;
    }
}
"""

theme = gr.themes.Soft(
    primary_hue=gr.themes.colors.slate,
    secondary_hue=gr.themes.colors.slate,
    neutral_hue=gr.themes.colors.slate,
).set(
    block_background_fill="*neutral_950",
    block_label_background_fill="*neutral_900",
    block_label_text_color="rgb(166, 173, 186)",
    button_primary_background_fill="rgb(75, 85, 99)",
    button_primary_background_fill_hover="rgb(55, 65, 79)",
    input_background_fill="rgb(17, 24, 39)",
    border_color_primary="rgb(55, 65, 79)",
)

with gr.Blocks(theme=theme, css=custom_css) as demo:
    gr.Markdown("# AI Research Assistant")
    with gr.Tabs():
        # ----- Chat Tab -----
        with gr.TabItem("Chat"):
            gr.Markdown(
                "Welcome! Ask your research question below. "
                "Select a model type and search provider as needed."
            )
            with gr.Row():
                model_choice = gr.Dropdown(
                    label="Model Type", choices=["Chat", "Structured"], value="Chat"
                )
                search_provider = gr.Dropdown(
                    label="Search Provider", choices=["DuckDuckGo", "Google"], value="DuckDuckGo"
                )
            with gr.Column():
                chatbot = gr.Chatbot(label="Conversation")
            with gr.Row():
                user_query = gr.Textbox(
                    placeholder="Ask your research question...",
                    lines=2,
                    show_label=False
                )
            with gr.Row():
                submit_btn = gr.Button("Search")
                clear_btn = gr.Button("Clear Chat")
                copy_btn = gr.Button("Copy Conversation")
            conversation_history = gr.State([])
            conv_text = gr.Textbox(label="Conversation Text (copy below)", interactive=False)
            submit_btn.click(
                fn=process_query,
                inputs=[user_query, conversation_history, model_choice, search_provider],
                outputs=[chatbot, conversation_history],
                show_progress=True
            )
            user_query.submit(
                fn=process_query,
                inputs=[user_query, conversation_history, model_choice, search_provider],
                outputs=[chatbot, conversation_history],
                show_progress=True
            )
            clear_btn.click(fn=clear_chat, inputs=[], outputs=[chatbot, conversation_history])
            copy_btn.click(fn=get_conversation_text, inputs=[conversation_history], outputs=[conv_text])
            
app = FastAPI()

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
