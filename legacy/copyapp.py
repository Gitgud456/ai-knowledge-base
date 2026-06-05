import os
import streamlit as st
from streamlit_chat import message # Unused, but keep for now if you plan to use it later
import networkx as nx
from streamlit_agraph import agraph, Node, Edge, Config
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import CrossEncoder
import re # Make sure 're' is imported for reparse_mentor_plan

# Import all utility functions
# IMPORTANT: Ensure these files (llm_utils.py, db_utils.py) are in the same directory
# or properly installed as a package.
from llm_utils import (
    init_history_db, list_sessions, create_new_session,
    get_session_history, add_message_to_session, delete_session,
    run_reflective_agent_loop, start_mentor_session, continue_mentor_session
)
from db_utils import (
    learn_from_document, learn_from_vault,
    manage_knowledge_list, manage_knowledge_delete, build_knowledge_graph,
    get_proactive_insights
)

# --- INITIALIZATION ---
st.set_page_config(page_title="AI Agent Assistant", layout="wide")

@st.cache_resource
def initialize_components():
    """Load heavy models and initialize clients only once."""
    print("--- Initializing AI Components ---")
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="BAAI/bge-large-en-v1.5")
    db_client = chromadb.PersistentClient(path="./chroma_db")
    collection = db_client.get_or_create_collection(name="knowledge_base", embedding_function=embedding_function)
    return cross_encoder, collection

cross_encoder, collection = initialize_components()
init_history_db()

# --- App Settings in Session State ---
if 'settings' not in st.session_state:
    st.session_state.settings = {
        "local_llm": "llama3:8b-instruct-q4_K_M",
        "n_results": 10,
        "top_k": 5
    }

# Ensure session IDs for both modes and active mode tracking
if 'current_chat_session_id' not in st.session_state:
    st.session_state.current_chat_session_id = None
if 'current_mentor_session_id' not in st.session_state:
    st.session_state.current_mentor_session_id = None
if 'active_mode' not in st.session_state:
    st.session_state.active_mode = "chat" # Default to chat mode

# Message histories are now tied to active session IDs
if 'agent_messages' not in st.session_state:
    st.session_state.agent_messages = []
if 'mentor_messages' not in st.session_state:
    st.session_state.mentor_messages = []

# Add state for mentor mode to track the plan and current step
if 'mentor_plan' not in st.session_state:
    st.session_state.mentor_plan = []
if 'current_topic_index' not in st.session_state:
    st.session_state.current_topic_index = 0

# Helper to stream and capture content
# This function is now mostly for the chat agent, mentor mode will handle streaming directly
def stream_and_capture(generator):
    full_content = ""
    for chunk in generator:
        full_content += chunk
        yield chunk
    st.session_state.last_streamed_content = full_content
    return

# --- Function to re-parse mentor plan from messages ---
def reparse_mentor_plan(messages):
    mentor_plan = []
    plan_text = ""
    
    st.write("--- DEBUG: reparse_mentor_plan ---")
    st.write(f"Messages count for parsing: {len(messages)}")

    # Search for the most recent assistant message that contains "LEARNING PLAN:" (case-insensitive)
    for msg in reversed(messages):
        if msg['role'] == 'assistant':
            plan_match = re.search(r'(?:^|\n)\s*([Ll][Ee][Aa][Rr][Nn][Ii][Nn][Gg]\s+[Pp][Ll][Aa][Nn]:)', msg['content'])
            if plan_match:
                plan_text = msg['content']
                st.write(f"Found '{plan_match.group(1)}' in assistant message. Content snippet:")
                st.write(plan_text[:500] + "..." if len(plan_text) > 500 else plan_text)
                break
            else:
                st.write(f"Assistant message does not contain 'LEARNING PLAN:' (case-insensitive). Content snippet: {msg['content'][:200]}...")


    if not plan_text:
        st.write("No explicit 'LEARNING PLAN:' marker found. Attempting fallback to first assistant message for plan.")
        
        user_mentor_request_found = False
        first_assistant_response_after_mentor_request = None
        for i, msg in enumerate(messages):
            if msg['role'] == 'user' and "I want to master the topic:" in msg['content']:
                user_mentor_request_found = True
            elif user_mentor_request_found and msg['role'] == 'assistant':
                first_assistant_response_after_mentor_request = msg['content']
                st.write("DEBUG: Found first assistant response after user mentor request for fallback.")
                break
        
        if first_assistant_response_after_mentor_request:
            plan_text = first_assistant_response_after_mentor_request
            st.write("DEBUG: Fallback plan_text populated from first assistant response.")
        else:
            st.write("DEBUG: No suitable assistant message found for fallback. Returning empty plan.")
            return []

    plan_start_index = -1
    match = re.search(r'([Ll][Ee][Aa][Rr][Nn][Ii][Nn][Gg]\s+[Pp][Ll][Aa][Nn]:)', plan_text)
    if match:
        plan_start_index = match.end()
        st.write(f"Successfully found plan start marker (case-insensitive) at index {plan_start_index}.")
    else:
        st.write("Could not find case-insensitive 'LEARNING PLAN:' marker for specific start index. Starting parsing from the beginning of the selected plan_text.")
        plan_start_index = 0

    content_after_plan_start = plan_text[plan_start_index:].strip()
    st.write("Content after potential 'LEARNING PLAN:' marker (or start of text):")
    st.write(content_after_plan_start[:500] + "..." if len(content_after_plan_start) > 500 else content_after_plan_start)

    end_markers = [
        "Let's start with the first topic.",
        "Let's dive into the first topic:",
        "We can begin with the first topic:",
        "Would you like to start with the first topic?",
        "If you have any questions before we begin, just ask!",
        "Feel free to ask any questions about the plan or the first topic.",
        "Ready to begin?",
        "Let me know when you're ready to start.",
        "YOUR DETAILED EXPLANATION OF THE FIRST TOPIC:", # Added to catch direct prompts from LLM utils
        "YOUR DETAILED EXPLANATION OF" # More general match
    ]
    
    actual_plan_content = content_after_plan_start
    found_end_marker = False
    for marker in end_markers:
        # CORRECTED LINE 1: Use 'content_after_plan_start' instead of 'content_after_plan_content'
        marker_match = re.search(re.escape(marker), content_after_plan_start, re.IGNORECASE)
        if marker_match:
            # CORRECTED LINE 2: Use 'content_after_plan_start' instead of 'content_after_plan_content'
            actual_plan_content = content_after_plan_start[:marker_match.start()].strip()
            found_end_marker = True
            st.write(f"Found end marker (case-insensitive): '{marker}' at index {marker_match.start()}. Actual plan content before marker:")
            st.write(actual_plan_content[:500] + "..." if len(actual_plan_content) > 500 else actual_plan_content)
            break

    if not found_end_marker:
        st.write("No specific end marker found. Taking all content after 'LEARNING PLAN:' as plan content.")

    lines_processed = 0
    for line in actual_plan_content.split('\n'):
        stripped_line = line.strip()
        if re.match(r'^\d+\.\s*(.+)', stripped_line):
            mentor_plan.append(stripped_line)
            st.write(f"Matched line (added to plan): '{stripped_line}'")
            lines_processed += 1
        else:
            st.write(f"Did not match line (skipped): '{stripped_line}'")
    
    if lines_processed == 0:
        st.write("WARNING: No numbered plan lines were extracted from the content. This could mean the LLM didn't format it as a numbered list, or the plan section was too small/not found correctly.")

    st.write(f"--- DEBUG: Final mentor_plan: {mentor_plan} ---")
    return mentor_plan

# --- Function to re-infer current topic index ---
def reinfer_current_topic_index(mentor_messages, mentor_plan):
    st.write("--- DEBUG: reinfer_current_topic_index ---")
    st.write(f"Plan for inferring: {mentor_plan}")
    if not mentor_plan:
        st.write("DEBUG: Mentor plan is empty, returning 0.")
        return 0

    # Iterate from the most recent messages backward
    for i in range(len(mentor_messages) - 1, -1, -1):
        msg_content = mentor_messages[i]['content']
        if mentor_messages[i]['role'] == 'assistant':
            st.write(f"DEBUG: Checking assistant message for topic inference (last 100 chars): {msg_content[-100:]}")
            for idx, topic_line in enumerate(mentor_plan):
                match = re.match(r'^\d+\.\s*(.+)', topic_line)
                if match:
                    topic_name = match.group(1).strip()
                    # Check if the topic name is strongly present in the message content
                    # Using a word boundary regex to avoid partial matches (e.g., "intro" matching "introduction")
                    if re.search(r'\b' + re.escape(topic_name) + r'\b', msg_content, re.IGNORECASE):
                        st.write(f"DEBUG: Found topic '{topic_name}' (index {idx}) in assistant message.")
                        return idx
    st.write("DEBUG: No topic match found in assistant messages. Returning 0.")
    return 0 # If no topic match, default to first topic or 0

# --- UI TABS ---
tab_names = ["💬 Chat Agent", "🎓 Mentor Mode", "🗺️ Knowledge Graph", "⚙️ Settings"]
current_tab_name = st.sidebar.radio("Navigation", tab_names, key="main_tabs_radio")

# Update active_mode based on selected tab and manage session history loading
if current_tab_name == "💬 Chat Agent":
    if st.session_state.active_mode != "chat":
        st.session_state.active_mode = "chat"
        st.session_state.current_mentor_session_id = None
        st.session_state.mentor_messages = []
        st.session_state.mentor_plan = []
        st.session_state.current_topic_index = 0
        if st.session_state.current_chat_session_id:
            st.session_state.agent_messages = get_session_history(st.session_state.current_chat_session_id)
        else:
            st.session_state.agent_messages = []
        st.write("DEBUG: Switched to Chat Agent mode.")
    
    st.header("Chat with your Self-Correcting AI Agent")
    
    if st.session_state.current_chat_session_id is None:
        st.info("Please start a new session or load a previous one from the sidebar to begin chatting.")
    else:
        for msg in st.session_state.agent_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        prompt = st.chat_input("Ask your AI agent...", key="chat_input_agent")
        
        if prompt:
            st.session_state.agent_messages.append({"role": "user", "content": prompt})
            add_message_to_session(st.session_state.current_chat_session_id, "user", prompt)
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("🧠 Agent is reasoning, acting, and reflecting..."):
                    response_generator = run_reflective_agent_loop(prompt, collection, cross_encoder, st.session_state.settings)
                    
                    st.write_stream(stream_and_capture(response_generator))
                    full_response_content = st.session_state.last_streamed_content

            st.session_state.agent_messages.append({"role": "assistant", "content": full_response_content})
            add_message_to_session(st.session_state.current_chat_session_id, "assistant", full_response_content)
            st.rerun()

elif current_tab_name == "🎓 Mentor Mode":
    # Ensure active mode is set correctly when switching to this tab
    if st.session_state.active_mode != "mentor":
        st.session_state.active_mode = "mentor"
        st.session_state.current_chat_session_id = None
        st.session_state.agent_messages = []
        st.write(f"DEBUG: Switched to Mentor Mode. current_mentor_session_id: {st.session_state.current_mentor_session_id}")
        # Load mentor messages if a session is currently selected
        if st.session_state.current_mentor_session_id:
            st.session_state.mentor_messages = get_session_history(st.session_state.current_mentor_session_id)
            st.write(f"DEBUG: Loaded mentor_messages for session {st.session_state.current_mentor_session_id}. Count: {len(st.session_state.mentor_messages)}")
            
            st.session_state.mentor_plan = reparse_mentor_plan(st.session_state.mentor_messages)
            st.write(f"DEBUG: After reparse_mentor_plan, mentor_plan: {st.session_state.mentor_plan}")
            
            st.session_state.current_topic_index = reinfer_current_topic_index(st.session_state.mentor_messages, st.session_state.mentor_plan)
            st.write(f"DEBUG: After reinfer_current_topic_index, current_topic_index: {st.session_state.current_topic_index}")
        else:
            st.session_state.mentor_messages = []
            st.session_state.mentor_plan = []
            st.session_state.current_topic_index = 0
            st.write("DEBUG: No current mentor session, initializing empty mentor state.")

    st.header("🎓 Mentor Mode")
    st.info("Enter a topic you want to master. The AI will create a structured learning plan based on your knowledge base and guide you through it.")
    
    # Conditional rendering for the mentor_topic input field
    if st.session_state.current_mentor_session_id is None:
        mentor_topic = st.text_input("What topic do you want to master today?", key="mentor_topic_initial_input") # Unique key
        
        if st.button("Start Mentoring Session", key="start_mentor_session_btn"):
            if mentor_topic:
                session_name = f"Mentor: {mentor_topic}"
                st.session_state.current_mentor_session_id = create_new_session(session_name)
                st.session_state.current_topic_index = 0
                st.session_state.mentor_plan = [] # Reset plan
                st.session_state.mentor_messages = [] # Reset messages
                
                user_message_content = f"I want to master the topic: {mentor_topic}"
                user_message = {"role": "user", "content": user_message_content}
                
                with st.chat_message("user", avatar="🎓"):
                    st.markdown(user_message["content"])
                
                st.session_state.mentor_messages.append(user_message)
                add_message_to_session(st.session_state.current_mentor_session_id, user_message["role"], user_message["content"])
                
                # --- MODIFIED BLOCK FOR INITIAL MENTOR RESPONSE ---
                placeholder = st.empty() # Create a placeholder for the streamed content

                with st.chat_message("assistant", avatar="🎓"):
                    with st.spinner("🧠 Building your learning plan..."):
                        st.write("DEBUG (app.py): Calling start_mentor_session to generate initial plan and topic...")
                        response_generator, _ = start_mentor_session(mentor_topic, collection, cross_encoder, st.session_state.settings)
                        
                        st.write("DEBUG (app.py): Received generator. Streaming and capturing full content...")
                        
                        full_response_content = ""
                        for chunk in response_generator:
                            full_response_content += chunk
                            # Update the placeholder with new content + blinking cursor
                            placeholder.markdown(full_response_content + "▌") 
                        # Display final content without cursor
                        placeholder.markdown(full_response_content) 

                        st.session_state.last_streamed_content = full_response_content # Store the full content
                        st.write(f"DEBUG (app.py): Captured full_response_content length: {len(full_response_content)}")
                # --- END MODIFIED BLOCK ---
                
                ai_message = {"role": "assistant", "content": full_response_content}
                st.session_state.mentor_messages.append(ai_message)
                add_message_to_session(st.session_state.current_mentor_session_id, ai_message["role"], ai_message["content"])
                
                st.write("DEBUG (app.py): Attempting to reparse plan from updated mentor_messages...")
                st.session_state.mentor_plan = reparse_mentor_plan(st.session_state.mentor_messages)
                
                st.rerun()

            else:
                st.warning("Please enter a topic to start mentoring.")
    
    else: # A mentor session is active, so display the messages and chat input
        # Display the topic being mastered at the top if a session is active
        if st.session_state.current_mentor_session_id:
            session_name_from_id = next((s['name'] for s in list_sessions() if s['id'] == st.session_state.current_mentor_session_id), "Current Session")
            if session_name_from_id.startswith("Mentor: "):
                st.subheader(f"Mastering: {session_name_from_id.replace('Mentor: ', '')}")

        # This loop displays the chat history
        for msg in st.session_state.mentor_messages:
            with st.chat_message(msg["role"], avatar="🎓" if msg["role"] == "assistant" else "user"):
                st.markdown(msg["content"])
        
        # This is the chat input for continuing the session
        followup_prompt = st.chat_input("Continue your mentoring session...", key="chat_input_mentor")
        
        if followup_prompt:
            st.write(f"DEBUG: User entered followup_prompt: '{followup_prompt}'")
            st.write(f"DEBUG: Current mentor_plan state: {st.session_state.mentor_plan}")
            st.write(f"DEBUG: Current current_topic_index: {st.session_state.current_topic_index}")

            if not st.session_state.mentor_plan:
                st.write("DEBUG: Mentor plan is empty before continue_mentor_session, attempting reparse.")
                st.session_state.mentor_plan = reparse_mentor_plan(st.session_state.mentor_messages)
                st.session_state.current_topic_index = reinfer_current_topic_index(st.session_state.mentor_messages, st.session_state.mentor_plan)
                st.write(f"DEBUG: After reparse in followup_prompt, mentor_plan: {st.session_state.mentor_plan}")
                st.write(f"DEBUG: After reparse in followup_prompt, current_topic_index: {st.session_state.current_topic_index}")
                
                if not st.session_state.mentor_plan:
                    st.warning("Could not re-establish the mentor plan from history. Please start a new session or check the plan format.", icon="⚠️")
                    st.stop()

            user_followup_message = {"role": "user", "content": followup_prompt}
            st.session_state.mentor_messages.append(user_followup_message)
            add_message_to_session(st.session_state.current_mentor_session_id, user_followup_message["role"], user_followup_message["content"])
            
            with st.chat_message("user"):
                st.markdown(user_followup_message["content"])

            with st.chat_message("assistant", avatar="🎓"):
                with st.spinner("Mentor is thinking..."):
                    st.write("DEBUG: Calling continue_mentor_session...")
                    response_generator, new_topic_index = continue_mentor_session(
                        followup_prompt, 
                        st.session_state.mentor_messages, 
                        st.session_state.mentor_plan, 
                        st.session_state.current_topic_index, 
                        collection, 
                        cross_encoder, 
                        st.session_state.settings
                    )
                    st.write(f"DEBUG: Received generator from continue_mentor_session. New topic index: {new_topic_index}")
                    
                    st.write_stream(stream_and_capture(response_generator))
                    full_response_content = st.session_state.last_streamed_content
                    st.write(f"DEBUG: Captured full_response_content length after continue_mentor_session: {len(full_response_content)}")
            
            ai_followup_message = {"role": "assistant", "content": full_response_content}
            st.session_state.mentor_messages.append(ai_followup_message)
            add_message_to_session(st.session_state.current_mentor_session_id, ai_followup_message["role"], ai_followup_message["content"])
            
            st.session_state.current_topic_index = new_topic_index # Update index after response
            st.write(f"DEBUG: Updated current_topic_index to {st.session_state.current_topic_index}")

            st.rerun()

elif current_tab_name == "🗺️ Knowledge Graph":
    st.header("Explore Your Knowledge Graph")
    graph_query = st.text_input("Enter a topic to explore its connections:", key="graph_query")
    if st.button("Generate Graph", key="generate_graph_btn"):
        if graph_query:
            with st.spinner("Building graph..."):
                G = build_knowledge_graph(collection, graph_query)
                if G and G.number_of_nodes() > 1:
                    nodes = [Node(id=n, label=G.nodes[n]['label'], **G.nodes[n]) for n in G.nodes]
                    edges = [Edge(source=u, target=v) for u, v in G.edges]
                    config = Config(width=1200, height=800, directed=False, physics=True, hierarchical=False)
                    agraph(nodes=nodes, edges=edges, config=config)
                else:
                    st.warning("Could not find enough connections for this topic.")
        else:
            st.warning("Please enter a topic.")

elif current_tab_name == "⚙️ Settings":
    st.header("Settings")
    st.write("Configure the AI's parameters. Changes will apply on the next interaction.")

    st.session_state.settings['local_llm'] = st.selectbox(
        "Local LLM Model",
        options=["llama3:8b-instruct-q4_K_M", "mistral", "llama3:latest"],
        index=["llama3:8b-instruct-q4_K_M", "mistral", "llama3:latest"].index(st.session_state.settings['local_llm']),
        key="llm_model_select"
    )
    st.session_state.settings['n_results'] = st.slider(
        "Documents to Retrieve (Recall)",
        min_value=5, max_value=25, value=st.session_state.settings['n_results'],
        key="n_results_slider"
    )
    st.session_state.settings['top_k'] = st.slider(
        "Documents to Use (Re-ranked)",
        min_value=2, max_value=10, value=st.session_state.settings['top_k'],
        key="top_k_slider"
    )
    st.success("Settings saved for this session.")

# --- SIDEBAR ---
with st.sidebar:
    st.divider()

    st.subheader("For You ✨")
    with st.spinner("Finding connections..."):
        insights = get_proactive_insights(collection)
    st.info(insights)

    st.divider()

    st.subheader("Session Management")
    new_session_name = st.text_input("New Session Name", key="new_session_name_input")
    if st.button("Start New Session", key="start_new_session_btn"):
        if new_session_name:
            if st.session_state.active_mode == "chat":
                st.session_state.current_chat_session_id = create_new_session(new_session_name)
                st.session_state.agent_messages = []
            elif st.session_state.active_mode == "mentor":
                st.session_state.current_mentor_session_id = create_new_session(f"Mentor: {new_session_name}")
                st.session_state.mentor_messages = []
                st.session_state.mentor_plan = []
                st.session_state.current_topic_index = 0
            
            # Reset the other mode's session state when starting a new session in the current mode
            if st.session_state.active_mode == "chat":
                st.session_state.current_mentor_session_id = None
                st.session_state.mentor_messages = []
                st.session_state.mentor_plan = []
                st.session_state.current_topic_index = 0
            elif st.session_state.active_mode == "mentor":
                st.session_state.current_chat_session_id = None
                st.session_state.agent_messages = []

            st.rerun()
        else:
            st.warning("Please enter a name.")

    saved_sessions = list_sessions()
    if saved_sessions:
        session_options = {f"{s['name']} (ID: {s['id']})": s['id'] for s in saved_sessions}
        selected_session_display = ""
        # Determine the currently selected session in the dropdown for correct default indexing
        if st.session_state.active_mode == "chat" and st.session_state.current_chat_session_id:
            for display, sid in session_options.items():
                if sid == st.session_state.current_chat_session_id:
                    selected_session_display = display
                    break
        elif st.session_state.active_mode == "mentor" and st.session_state.current_mentor_session_id:
            for display, sid in session_options.items():
                if sid == st.session_state.current_mentor_session_id:
                    selected_session_display = display
                    break
        
        try:
            # Add 1 to index because of the initial "" option
            default_index = list(session_options.keys()).index(selected_session_display) + 1 if selected_session_display else 0
        except ValueError:
            default_index = 0 # Default to empty if not found

        selected_session_str = st.selectbox("Load/Delete Sessions", options=[""] + list(session_options.keys()), 
                                             index=default_index, key="session_select")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Load Session", key="load_session_btn"):
                if selected_session_str:
                    session_id_to_load = session_options.get(selected_session_str)
                    session_name_to_load = selected_session_str.split(' (ID:')[0]

                    if session_name_to_load.startswith("Mentor:"):
                        st.session_state.current_mentor_session_id = session_id_to_load
                        st.session_state.current_chat_session_id = None # Deactivate chat session
                        st.session_state.mentor_messages = get_session_history(session_id_to_load)
                        st.session_state.agent_messages = [] # Clear agent messages
                        st.session_state.active_mode = "mentor"
                        
                        st.write(f"DEBUG Sidebar: Loading mentor session {session_id_to_load}. Messages count: {len(st.session_state.mentor_messages)}")
                        st.session_state.mentor_plan = reparse_mentor_plan(st.session_state.mentor_messages)
                        st.session_state.current_topic_index = reinfer_current_topic_index(st.session_state.mentor_messages, st.session_state.mentor_plan)
                        st.write(f"DEBUG Sidebar: After load, mentor_plan: {st.session_state.mentor_plan}")
                        st.write(f"DEBUG Sidebar: After load, current_topic_index: {st.session_state.current_topic_index}")
                                                            
                    else: # It's a chat session
                        st.session_state.current_chat_session_id = session_id_to_load
                        st.session_state.current_mentor_session_id = None # Deactivate mentor session
                        st.session_state.agent_messages = get_session_history(session_id_to_load)
                        st.session_state.mentor_messages = [] # Clear mentor messages
                        st.session_state.mentor_plan = []
                        st.session_state.current_topic_index = 0
                        st.session_state.active_mode = "chat"
                        st.write(f"DEBUG Sidebar: Loading chat session {session_id_to_load}.")
                    st.rerun()

        with col2:
            if st.button("Delete", type="primary", key="delete_session_btn"):
                if selected_session_str:
                    session_id_to_delete = session_options.get(selected_session_str)
                    delete_session(session_id_to_delete)
                    if st.session_state.current_chat_session_id == session_id_to_delete:
                        st.session_state.current_chat_session_id = None
                        st.session_state.agent_messages = []
                    if st.session_state.current_mentor_session_id == session_id_to_delete:
                        st.session_state.current_mentor_session_id = None
                        st.session_state.mentor_messages = []
                        st.session_state.mentor_plan = []
                        st.session_state.current_topic_index = 0
                    st.rerun()

    st.divider()

    st.subheader("Knowledge Base")
    uploaded_file = st.file_uploader("Learn from a single document", type=['pdf', 'txt', 'epub'], key="file_uploader")
    if uploaded_file is not None:
        temp_path = f"./temp_{uploaded_file.name}";
        with open(temp_path, "wb") as f: f.write(uploaded_file.getvalue())
        with st.spinner("Learning from document..."):
            learn_from_document(collection, temp_path, st)
        os.remove(temp_path)

    st.subheader("Obsidian Vault")
    if st.button("Sync Vault", key="sync_vault_btn"):
        with st.spinner("Syncing with your Obsidian Vault..."):
            learn_from_vault(collection, st)

    st.divider()

    if st.button("List Learned Sources", key="list_sources_btn"):
        st.info(manage_knowledge_list(collection))
    source_to_delete = st.text_input("Source Name to Delete", key="source_delete_input")
    if st.button("Delete Source", key="delete_source_btn"):
        with st.spinner("Deleting..."):
            st.success(manage_knowledge_delete(collection, source_to_delete))