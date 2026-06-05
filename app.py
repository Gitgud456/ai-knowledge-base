import os
import streamlit as st
import networkx as nx
from streamlit_agraph import agraph, Node, Edge, Config
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import CrossEncoder
import re

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
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="BAAI/bge-large-en-v1.5")
    db_client = chromadb.PersistentClient(path="./chroma_db")
    collection = db_client.get_or_create_collection(name="knowledge_base", embedding_function=embedding_function)
    return cross_encoder, collection

cross_encoder, collection = initialize_components()
init_history_db()

# --- App Settings & State Management ---
if 'settings' not in st.session_state:
    st.session_state.settings = {"local_llm": "llama3:8b-instruct-q4_K_M", "n_results": 10, "top_k": 5}
if 'current_chat_session_id' not in st.session_state: st.session_state.current_chat_session_id = None
if 'current_mentor_session_id' not in st.session_state: st.session_state.current_mentor_session_id = None
if 'agent_messages' not in st.session_state: st.session_state.agent_messages = []
if 'mentor_messages' not in st.session_state: st.session_state.mentor_messages = []
if 'mentor_plan' not in st.session_state: st.session_state.mentor_plan = []
if 'current_topic_index' not in st.session_state: st.session_state.current_topic_index = 0

def parse_plan_from_text(text):
    """A robust function to extract a numbered list from the AI's plan response."""
    plan_items = []
    plan_match = re.search(r'LEARNING PLAN:(.*)', text, re.DOTALL | re.IGNORECASE)
    if plan_match:
        plan_text = plan_match.group(1)
        plan_items = re.findall(r'^\s*\d+\.\s*(.*)', plan_text, re.MULTILINE)
    return [item.strip() for item in plan_items]

# --- UI TABS ---
# Using tabs is simpler for state than radio buttons
selected_tab = st.sidebar.selectbox("Mode", ["Chat Agent", "Mentor Mode", "Knowledge Graph", "Settings"])

if selected_tab == "Chat Agent":
    st.header("Chat with your Self-Correcting AI Agent")
    if st.session_state.current_chat_session_id is None:
        st.info("Please start or load a Chat session from the sidebar.")
    else:
        # Load messages from DB for the active session
        st.session_state.agent_messages = get_session_history(st.session_state.current_chat_session_id)
        for msg in st.session_state.agent_messages:
            with st.chat_message(msg["role"]): st.markdown(msg["content"])

        if prompt := st.chat_input("Ask your AI agent..."):
            add_message_to_session(st.session_state.current_chat_session_id, "user", prompt)
            with st.chat_message("user"): st.markdown(prompt)
            with st.chat_message("assistant"):
                with st.spinner("🧠 Agent is reasoning..."):
                    response_generator = run_reflective_agent_loop(prompt, collection, cross_encoder, st.session_state.settings)
                    response = st.write_stream(response_generator)
            add_message_to_session(st.session_state.current_chat_session_id, "assistant", response)
            st.rerun()

elif selected_tab == "Mentor Mode":
    st.header("🎓 Mentor Mode")
    if st.session_state.current_mentor_session_id is None:
        st.info("Start a new mentoring session from the sidebar.")
        mentor_topic = st.text_input("What topic do you want to master today?", key="mentor_topic_initial")
        if st.button("Start Mentoring Session"):
            if mentor_topic:
                session_name = f"Mentor: {mentor_topic}"
                st.session_state.current_mentor_session_id = create_new_session(session_name)
                # Clear previous mentor state
                st.session_state.mentor_messages = []
                st.session_state.mentor_plan = []
                st.session_state.current_topic_index = 0
                # Add first user message
                add_message_to_session(st.session_state.current_mentor_session_id, "user", f"I want to master the topic: {mentor_topic}")
                st.rerun() # Rerun to enter the active session view
    else:
        # Load and display active mentor session
        st.session_state.mentor_messages = get_session_history(st.session_state.current_mentor_session_id)
        
        # If plan is not in state but history exists, create it
        if not st.session_state.mentor_plan and len(st.session_state.mentor_messages) > 1:
             st.session_state.mentor_plan = parse_plan_from_text(st.session_state.mentor_messages[1]['content'])
        
        # Display plan if it exists
        if st.session_state.mentor_plan:
            st.markdown("**Your Learning Plan:**")
            for i, topic in enumerate(st.session_state.mentor_plan):
                if i == st.session_state.current_topic_index: st.markdown(f"**> {i+1}. {topic} (Current)**")
                else: st.markdown(f"&nbsp;&nbsp;&nbsp;{i+1}. {topic}")
            st.divider()

        # Display full chat history
        for msg in st.session_state.mentor_messages:
            with st.chat_message(msg["role"], avatar="🎓" if msg["role"] == "assistant" else "user"): st.markdown(msg["content"])
        
        # First-time plan generation
        if len(st.session_state.mentor_messages) == 1:
            with st.chat_message("assistant", avatar="🎓"):
                with st.spinner("🧠 Building your learning plan..."):
                    response_generator, get_full_text = start_mentor_session(st.session_state.mentor_messages[0]['content'], collection, cross_encoder, st.session_state.settings)
                    response = st.write_stream(response_generator)
            full_response_content = get_full_text()
            add_message_to_session(st.session_state.current_mentor_session_id, "assistant", full_response_content)
            st.session_state.mentor_plan = parse_plan_from_text(full_response_content)
            st.session_state.current_topic_index = 0
            st.rerun()

        # Follow-up prompts
        if followup_prompt := st.chat_input("Continue your mentoring session..."):
            add_message_to_session(st.session_state.current_mentor_session_id, "user", followup_prompt)
            with st.chat_message("user"): st.markdown(followup_prompt)
            with st.chat_message("assistant", avatar="🎓"):
                with st.spinner("Mentor is thinking..."):
                    response_generator, new_index = continue_mentor_session(
                        followup_prompt, st.session_state.mentor_messages, st.session_state.mentor_plan,
                        st.session_state.current_topic_index, collection, cross_encoder, st.session_state.settings
                    )
                    response = st.write_stream(response_generator)
            add_message_to_session(st.session_state.current_mentor_session_id, "assistant", response)
            st.session_state.current_topic_index = new_index
            st.rerun()

elif selected_tab == "Knowledge Graph":
    # ... (Unchanged from your version)
    st.header("Explore Your Knowledge Graph")
    graph_query = st.text_input("Enter a topic to explore its connections:", key="graph_query")
    if st.button("Generate Graph"):
        if graph_query:
            with st.spinner("Building graph..."):
                G = build_knowledge_graph(collection, graph_query)
                if G and G.number_of_nodes() > 1:
                    nodes = [Node(id=n, label=G.nodes[n]['label'], **G.nodes[n]) for n in G.nodes]
                    edges = [Edge(source=u, target=v) for u, v in G.edges]
                    config = Config(width=1200, height=800, directed=False, physics=True, hierarchical=False)
                    agraph(nodes=nodes, edges=edges, config=config)
                else: st.warning("Could not find enough connections.")
        else: st.warning("Please enter a topic.")

elif selected_tab == "Settings":
    # ... (Unchanged from your version)
    st.header("Settings")
    st.session_state.settings['local_llm'] = st.selectbox("Local LLM Model", options=["llama3:8b-instruct-q4_K_M", "mistral", "llama3:latest"], index=["llama3:8b-instruct-q4_K_M", "mistral", "llama3:latest"].index(st.session_state.settings['local_llm']))
    st.session_state.settings['n_results'] = st.slider("Documents to Retrieve", min_value=5, max_value=25, value=st.session_state.settings['n_results'])
    st.session_state.settings['top_k'] = st.slider("Documents to Use", min_value=2, max_value=10, value=st.session_state.settings['top_k'])
    st.success("Settings saved for this session.")


# --- SIDEBAR ---
with st.sidebar:
    st.header("Control Panel")
    st.subheader("For You ✨")
    with st.spinner("Finding connections..."): st.info(get_proactive_insights(collection))
    st.divider()

    st.subheader("Session Management")
    new_session_name = st.text_input("New Session Name", key="new_session_input")
    if st.button("Start New Session"):
        if new_session_name:
            if selected_tab == "Mentor Mode":
                st.session_state.current_mentor_session_id = create_new_session(f"Mentor: {new_session_name}")
                st.session_state.mentor_messages = []
                st.session_state.mentor_plan = []
                st.session_state.current_topic_index = 0
            else: # Default to Chat
                st.session_state.current_chat_session_id = create_new_session(new_session_name)
                st.session_state.agent_messages = []
            st.rerun()
        else: st.warning("Please enter a name.")

    saved_sessions = list_sessions()
    if saved_sessions:
        session_options = {f"{s['name']} (ID: {s['id']})": s['id'] for s in saved_sessions}
        selected_session_str = st.selectbox("Load/Delete Sessions", options=[""] + list(session_options.keys()))
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Load"):
                if selected_session_str:
                    session_id = session_options[selected_session_str]
                    session_name = selected_session_str.split(' (ID:')[0]
                    # Reset state before loading
                    st.session_state.agent_messages, st.session_state.mentor_messages = [], []
                    st.session_state.mentor_plan, st.session_state.current_topic_index = [], 0
                    
                    if session_name.startswith("Mentor:"):
                        st.session_state.current_mentor_session_id = session_id
                        st.session_state.current_chat_session_id = None
                    else:
                        st.session_state.current_chat_session_id = session_id
                        st.session_state.current_mentor_session_id = None
                    st.rerun()
        with col2:
            if st.button("Delete", type="primary"):
                if selected_session_str:
                    session_id = session_options[selected_session_str]
                    delete_session(session_id)
                    if st.session_state.current_chat_session_id == session_id: st.session_state.current_chat_session_id = None
                    if st.session_state.current_mentor_session_id == session_id: st.session_state.current_mentor_session_id = None
                    st.rerun()

    st.divider()
    st.subheader("Knowledge Base")
    uploaded_file = st.file_uploader("Learn from a single document", type=['pdf', 'txt', 'epub'])
    if uploaded_file is not None:
        temp_path = f"./temp_{uploaded_file.name}"
        with open(temp_path, "wb") as f: f.write(uploaded_file.getvalue())
        with st.spinner("Learning..."): st.success(learn_from_document(collection, temp_path, st))
        os.remove(temp_path)

    st.subheader("Obsidian Vault")
    if st.button("Sync Vault"):
        with st.spinner("Syncing..."): st.success(learn_from_vault(collection, st))
    st.divider()

    if st.button("List Learned Sources"): st.info(manage_knowledge_list(collection))
    source_to_delete = st.text_input("Source Name to Delete")
    if st.button("Delete Source"):
        with st.spinner("Deleting..."): st.success(manage_knowledge_delete(collection, source_to_delete))