import os
import sqlite3
from datetime import datetime
import ollama
import json
import re
from dotenv import load_dotenv
from duckduckgo_search import DDGS

# --- SESSION HISTORY DATABASE (Unchanged) ---
DB_FILE = "session_history.db"
def init_history_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS sessions (id INTEGER PRIMARY KEY, name TEXT, created_at TEXT)")
    cursor.execute("CREATE TABLE IF NOT EXISTS messages (id INTEGER PRIMARY KEY, session_id INTEGER, role TEXT, content TEXT, timestamp TEXT, FOREIGN KEY (session_id) REFERENCES sessions (id) ON DELETE CASCADE)")
    conn.commit()
    conn.close()

def create_new_session(name):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    timestamp = datetime.now().isoformat()
    cursor.execute("INSERT INTO sessions (name, created_at) VALUES (?, ?)", (name, timestamp))
    session_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return session_id

def add_message_to_session(session_id, role, content):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    timestamp = datetime.now().isoformat()
    cursor.execute("INSERT INTO messages (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)", (session_id, role, content, timestamp))
    conn.commit()
    conn.close()

def get_session_history(session_id):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT role, content FROM messages WHERE session_id = ? ORDER BY timestamp ASC", (session_id,))
    history = [{"role": row[0], "content": row[1]} for row in cursor.fetchall()]
    conn.close()
    return history

def list_sessions():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, created_at FROM sessions ORDER BY created_at DESC")
    sessions = [{"id": row[0], "name": row[1], "created_at": row[2]} for row in cursor.fetchall()]
    conn.close()
    return sessions
    
def delete_session(session_id):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("PRAGMA foreign_keys = ON")
    cursor.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
    conn.commit()
    conn.close()
    return f"✅ Session {session_id} deleted."


# --- AGENT & LLM LOGIC (UPGRADED WITH MULTI-QUERY) ---

def search_knowledge_base(query, collection, cross_encoder, settings):
    """Tool to search the local knowledge base."""
    from db_utils import get_reranked_context
    print(f"Tool Used: search_knowledge_base, Query: {query}")
    context_docs = get_reranked_context(
        collection, cross_encoder, query, 
        n_results=settings['n_results'], top_k=settings['top_k']
    )
    # Return empty string instead of message to keep context clean
    return "\n---\n".join(context_docs) if context_docs else ""

def search_web(query):
    """Tool to search the web for real-time information."""
    print(f"Tool Used: search_web, Query: {query}")
    try:
        with DDGS() as ddgs:
            results = [r['body'] for r in ddgs.text(query, max_results=3)]
            return "\n---\n".join(results) if results else ""
    except Exception as e:
        return f"Web search failed: {e}"

def generate_streaming_response(messages, model):
    """A generator function that yields tokens for a streaming response."""
    try:
        stream = ollama.chat(model=model, messages=messages, stream=True)
        for chunk in stream:
            if 'message' in chunk and 'content' in chunk['message']:
                yield str(chunk['message']['content'])
    except Exception as e:
        yield f"❌ An error occurred with the local Ollama model: {e}"

def run_reflective_agent_loop(query, collection, cross_encoder, settings):
    """Implements the AI Agent with Multi-Query and Self-Correction."""
    
    # --- NEW: Step 1: Decompose the query ---
    decomposition_prompt = f"""You are a query analysis agent. Decompose the following user query into 1 to 3 simple, self-contained sub-queries.
This helps retrieve more accurate information. If the query is already simple, just return it as a single-item list.
Respond with ONLY a JSON object containing a single key "queries" which is a list of strings.

User Query: "{query}"
"""
    try:
        response = ollama.generate(model=settings['local_llm'], prompt=decomposition_prompt, format="json")
        sub_queries = json.loads(response['response']).get("queries", [query])
        print(f"Decomposed into sub-queries: {sub_queries}")
    except Exception as e:
        print(f"Query decomposition failed: {e}. Using original query.")
        sub_queries = [query]

    # --- Step 2: Run tools for each sub-query and gather context ---
    all_tool_results = []
    for sub_q in sub_queries:
        tool_prompt = f"""You are a reasoning agent. Based on the sub-query, which tool is best: 'search_knowledge_base' or 'search_web'?
Respond with ONLY a JSON object with "tool" and "query" keys.

Sub-query: "{sub_q}"
"""
        try:
            response = ollama.generate(model=settings['local_llm'], prompt=tool_prompt, format="json")
            tool_choice = json.loads(response['response'])
            tool_name = tool_choice.get("tool")
            tool_query = tool_choice.get("query")
            
            if tool_name == "search_knowledge_base":
                tool_result = search_knowledge_base(tool_query, collection, cross_encoder, settings)
                all_tool_results.append(tool_result)
            elif tool_name == "search_web":
                tool_result = search_web(tool_query)
                all_tool_results.append(tool_result)
        except Exception as e:
            print(f"Agent reasoning error for sub-query '{sub_q}': {e}")
            all_tool_results.append(search_knowledge_base(sub_q, collection, cross_encoder, settings))

    # Combine all results into a single context
    final_context = "\n\n".join(filter(None, all_tool_results))
    if not final_context.strip():
        final_context = "No relevant information was found for the query."

    # --- Step 3 & 4: Reflection and Final Answer (Unchanged) ---
    draft_prompt = f"CONTEXT:\n{final_context}\n\nUSER'S ORIGINAL QUERY:\n{query}\n\nDRAFT ANSWER:"
    draft_response = ollama.chat(model=settings['local_llm'], messages=[{'role': 'user', 'content': draft_prompt}])
    draft_answer = draft_response['message']['content']
    
    reflection_prompt = f"""You are a critique agent. Your goal is to improve a draft answer based on the provided context.
- Is the draft answer fully supported by the context? Does it miss any key details?
- Is it clear, concise, and directly answering the user's original question?
Based on your critique, provide a final, improved answer.

CONTEXT:
{final_context}

USER'S ORIGINAL QUERY:
{query}

DRAFT ANSWER:
{draft_answer}

FINAL, IMPROVED ANSWER:"""

    final_messages = [{"role": "user", "content": reflection_prompt}]
    return generate_streaming_response(final_messages, settings['local_llm'])


# --- MENTOR MODE LOGIC (Unchanged from your version) ---
def start_mentor_session(query, collection, cross_encoder, settings):
    """Generates a learning plan and the first lesson."""
    from db_utils import get_reranked_context
    context_docs = get_reranked_context(collection, cross_encoder, query, n_results=20, top_k=10)
    if not context_docs:
        def no_info_generator(): yield "I don't have enough information to build a learning plan."
        return no_info_generator(), []
    context_str = "\n---\n".join(context_docs)
    plan_prompt = f"""You are a master teacher... create a step-by-step learning plan...
After the plan, immediately provide a detailed explanation of the VERY FIRST topic...
CONTEXT:\n{context_str}\n"""
    messages_for_llm = [{"role": "user", "content": plan_prompt}]
    
    full_response_generator = generate_streaming_response(messages_for_llm, settings['local_llm'])
    
    full_response_content = ""
    def response_wrapper(generator):
        nonlocal full_response_content
        for chunk in generator:
            full_response_content += chunk
            yield chunk

    return response_wrapper(full_response_generator), lambda: full_response_content

def continue_mentor_session(user_query, chat_history, mentor_plan, current_topic_index, collection, cross_encoder, settings):
    """Continues a mentoring session by advancing the plan or answering a question."""
    from db_utils import get_reranked_context
    
    next_topic_commands = ["next", "proceed", "continue", "move on"]
    is_next_topic_command = any(cmd in user_query.lower() for cmd in next_topic_commands)
    
    new_topic_index = current_topic_index
    if is_next_topic_command:
        new_topic_index = current_topic_index + 1

    if is_next_topic_command and new_topic_index < len(mentor_plan):
        current_topic = mentor_plan[new_topic_index]
        context_docs = get_reranked_context(collection, cross_encoder, current_topic, n_results=10, top_k=5)
        context_str = "\n---\n".join(context_docs) if context_docs else "No specific context found."
        
        lesson_prompt = f"""The student is ready for the next topic: "{current_topic}"...
Please provide a comprehensive explanation...
CONTEXT:\n{context_str}\n"""
        messages = chat_history[-6:] + [{"role": "user", "content": lesson_prompt}]
        return generate_streaming_response(messages, settings['local_llm']), new_topic_index
        
    elif is_next_topic_command:
        def completion_generator(): yield "You've completed the learning plan! Great job!"
        return completion_generator(), new_topic_index
        
    else:
        context_docs = get_reranked_context(collection, cross_encoder, user_query, n_results=10, top_k=5)
        context_str = "\n---\n".join(context_docs) if context_docs else "No specific context found."
        
        qa_prompt = f"""The user has a follow-up question...
The current topic is "{mentor_plan[current_topic_index]}".
Answer their question thoroughly...
CONTEXT:\n{context_str}\n"""
        messages = chat_history[-6:] + [{"role": "user", "content": qa_prompt}]
        return generate_streaming_response(messages, settings['local_llm']), new_topic_index