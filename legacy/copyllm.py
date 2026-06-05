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

# ... (create_new_session, add_message_to_session, etc. are unchanged)
def create_new_session(name): conn = sqlite3.connect(DB_FILE); cursor = conn.cursor(); timestamp = datetime.now().isoformat(); cursor.execute("INSERT INTO sessions (name, created_at) VALUES (?, ?)", (name, timestamp)); session_id = cursor.lastrowid; conn.commit(); conn.close(); return session_id
def add_message_to_session(session_id, role, content): conn = sqlite3.connect(DB_FILE); cursor = conn.cursor(); timestamp = datetime.now().isoformat(); cursor.execute("INSERT INTO messages (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)", (session_id, role, content, timestamp)); conn.commit(); conn.close()
def get_session_history(session_id): conn = sqlite3.connect(DB_FILE); cursor = conn.cursor(); cursor.execute("SELECT role, content FROM messages WHERE session_id = ? ORDER BY timestamp ASC", (session_id,)); history = [{"role": row[0], "content": row[1]} for row in cursor.fetchall()]; conn.close(); return history
def list_sessions(): conn = sqlite3.connect(DB_FILE); cursor = conn.cursor(); cursor.execute("SELECT id, name, created_at FROM sessions ORDER BY created_at DESC"); sessions = [{"id": row[0], "name": row[1], "created_at": row[2]} for row in cursor.fetchall()]; conn.close(); return sessions
def delete_session(session_id): conn = sqlite3.connect(DB_FILE); cursor = conn.cursor(); cursor.execute("PRAGMA foreign_keys = ON"); cursor.execute("DELETE FROM sessions WHERE id = ?", (session_id,)); conn.commit(); conn.close(); return f"✅ Session {session_id} deleted."


# --- AGENT & LLM LOGIC (Unchanged) ---
def search_knowledge_base(query, collection, cross_encoder, settings):
    from db_utils import get_reranked_context
    context_docs = get_reranked_context(
        collection, cross_encoder, query, 
        n_results=settings['n_results'], top_k=settings['top_k']
    )
    return "\n---\n".join(context_docs) if context_docs else "No relevant information found."
# ... (rest of agent logic is unchanged)
def search_web(query):
    try:
        with DDGS() as ddgs:
            results = [r['body'] for r in ddgs.text(query, max_results=3)]
            return "\n---\n".join(results) if results else "No information found on the web."
    except Exception as e: return f"Web search failed: {e}"
def generate_streaming_response(messages, model):
    try:
        stream = ollama.chat(model=model, messages=messages, stream=True)
        for chunk in stream:
            if 'message' in chunk and 'content' in chunk['message']: yield str(chunk['message']['content'])
    except Exception as e: yield f"❌ An error occurred with the local Ollama model: {e}"
def run_reflective_agent_loop(query, collection, cross_encoder, settings):
    tool_prompt = f"""You are a reasoning agent... Respond with ONLY a JSON object...\nUser Query: "{query}" """
    try:
        response = ollama.generate(model=settings['local_llm'], prompt=tool_prompt, format="json")
        tool_choice = json.loads(response['response'])
        tool_name, tool_query = tool_choice.get("tool"), tool_choice.get("query")
        if tool_name == "search_knowledge_base": tool_result = search_knowledge_base(tool_query, collection, cross_encoder, settings)
        elif tool_name == "search_web": tool_result = search_web(tool_query)
        else: tool_result = "Invalid tool chosen."
    except Exception as e:
        print(f"Agent reasoning error: {e}.")
        tool_result = search_knowledge_base(query, collection, cross_encoder, settings)
    draft_prompt = f"CONTEXT:\n{tool_result}\n\nUSER'S ORIGINAL QUERY:\n{query}\n\nDRAFT ANSWER:"
    draft_response = ollama.chat(model=settings['local_llm'], messages=[{'role': 'user', 'content': draft_prompt}])
    draft_answer = draft_response['message']['content']
    reflection_prompt = f"""... critique ... a final, improved answer...\n\nCONTEXT:\n{tool_result}\n\nUSER'S ORIGINAL QUERY:\n{query}\n\nDRAFT ANSWER:\n{draft_answer}\n\nFINAL, IMPROVED ANSWER:"""
    final_messages = [{"role": "user", "content": reflection_prompt}]
    return generate_streaming_response(final_messages, settings['local_llm'])


# --- MENTOR MODE LOGIC (REVISED & CORRECTED) ---
def start_mentor_session(query, collection, cross_encoder, settings):
    """Generates a learning plan and the first lesson."""
    from db_utils import get_reranked_context
    context_docs = get_reranked_context(collection, cross_encoder, query, n_results=20, top_k=10)
    if not context_docs:
        def no_info_generator(): yield "I don't have enough information to build a learning plan."
        return no_info_generator(), []
    context_str = "\n---\n".join(context_docs)
    plan_prompt = f"""You are a master teacher. Based on the provided context, create a step-by-step learning plan for the user's topic: '{query}'.
List the key sub-topics as a numbered list, clearly labeled "LEARNING PLAN:".
After the plan, immediately provide a detailed explanation of the VERY FIRST topic.
At the end of the explanation, ask the user if they're ready for the next topic or have questions.

CONTEXT:
{context_str}
"""
    messages_for_llm = [{"role": "user", "content": plan_prompt}]
    
    # Capture the full response to parse the plan from it
    full_response_content = ""
    def response_wrapper(generator):
        nonlocal full_response_content
        for chunk in generator:
            full_response_content += chunk
            yield chunk

    stream_generator = generate_streaming_response(messages_for_llm, settings['local_llm'])
    return response_wrapper(stream_generator), lambda: full_response_content

def continue_mentor_session(user_query, chat_history, mentor_plan, current_topic_index, collection, cross_encoder, settings):
    """Continues a mentoring session, intelligently advancing the plan or answering questions."""
    from db_utils import get_reranked_context
    
    # Analyze the user's intent: Are they asking to move on?
    intent_prompt = f"""Analyze the user's last message in the context of a mentoring session.
The user might ask to 'proceed', say 'next', 'continue', or ask a question.
If the user's intent is to move to the next topic, respond with the single word "NEXT".
Otherwise, respond with "QUESTION".

Last user message: "{user_query}"
"""
    intent_response = ollama.generate(model=settings['local_llm'], prompt=intent_prompt)
    is_next_topic_command = "next" in intent_response['response'].lower()
    
    new_topic_index = current_topic_index
    if is_next_topic_command:
        new_topic_index = current_topic_index + 1

    if is_next_topic_command and new_topic_index < len(mentor_plan):
        # User wants the next topic in the plan
        current_topic = mentor_plan[new_topic_index]
        context_docs = get_reranked_context(collection, cross_encoder, current_topic, n_results=10, top_k=5)
        context_str = "\n---\n".join(context_docs) if context_docs else "No specific context found."
        
        lesson_prompt = f"""The student is ready for the next topic.
The current topic is: "{current_topic}"
Please provide a comprehensive and detailed explanation of this topic, using the provided context.
At the end, ask if they have questions or are ready to proceed to the next topic.

CONTEXT:
{context_str}
"""
        messages = chat_history[-6:] + [{"role": "user", "content": lesson_prompt}]
        return generate_streaming_response(messages, settings['local_llm']), new_topic_index
        
    elif is_next_topic_command:
        # User wants to proceed but is at the end of the plan
        def completion_generator(): yield "You've completed all topics in this learning plan! Great job! Feel free to ask any final questions or start a new mentor session."
        return completion_generator(), new_topic_index
        
    else:
        # User is asking a follow-up question
        context_docs = get_reranked_context(collection, cross_encoder, user_query, n_results=10, top_k=5)
        context_str = "\n---\n".join(context_docs) if context_docs else "No specific context found."
        
        qa_prompt = f"""The user has a follow-up question during a mentoring session.
The current topic is "{mentor_plan[current_topic_index]}".
Answer their question thoroughly using the provided context and conversation history.
At the end, prompt them to ask another question or continue to the next topic.

CONTEXT:
{context_str}
"""
        messages = chat_history[-6:] + [{"role": "user", "content": qa_prompt}]
        return generate_streaming_response(messages, settings['local_llm']), new_topic_index