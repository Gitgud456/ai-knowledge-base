import os
from datetime import datetime
import networkx as nx
from file_processor import extract_text_from_file, chunk_text
from obsidian_utils import find_markdown_files, read_obsidian_note
import streamlit as st

def learn_from_document(collection, file_path, st_object):
    """Processes a single document and adds its content to the knowledge base with a granular progress bar."""
    source_name = os.path.basename(file_path)

    if collection.get(where={"source": source_name})['ids']:
        st_object.warning(f"⚠️ This document ('{source_name}') already exists. To re-learn, please manage and delete it first.")
        return

    text = extract_text_from_file(file_path)
    if not text.strip():
        st_object.warning(f"Skipped '{source_name}': No extractable text found.")
        return

    chunks = chunk_text(text)
    if not chunks:
        st_object.warning(f"Skipped '{source_name}': No chunks generated.")
        return

    # --- Start progress bar for the entire ingestion process ---
    total_chunks = len(chunks)
    st_object.info(f"Ingesting {total_chunks} chunks from '{source_name}' to knowledge base...")
    progress_bar = st_object.progress(0, text="Starting ingestion...")

    BATCH_SIZE = 100 # Adjust batch size based on performance
    
    for i in range(0, total_chunks, BATCH_SIZE):
        batch_chunks = chunks[i:i + BATCH_SIZE]
        batch_metadatas = [{"source": source_name, "learned_at": datetime.now().isoformat()} for _ in batch_chunks]
        batch_ids = [f"{source_name}_{j}" for j in range(i, i + len(batch_chunks))]

        try:
            collection.add(documents=batch_chunks, metadatas=batch_metadatas, ids=batch_ids)
        except Exception as e:
            st_object.error(f"❌ Error adding batch {i}-{i+len(batch_chunks)} from '{source_name}' to ChromaDB: {e}")
            progress_bar.empty() # Clear the progress bar on error
            return

        # Update progress bar
        current_progress = min((i + len(batch_chunks)) / total_chunks, 1.0)
        progress_percent = int(current_progress * 100)
        progress_bar.progress(progress_percent, text=f"Ingesting chunks: {i + len(batch_chunks)}/{total_chunks}")
    
    progress_bar.empty() # Clear the progress bar after completion
    st_object.success(f"✅ Successfully learned from '{source_name}'. Added {total_chunks} chunks.")


def learn_from_vault(collection, st_object):
    """Finds all notes in an Obsidian vault and learns from them with granular progress bars."""
    vault_path = os.getenv("OBSIDIAN_VAULT_PATH")
    if not vault_path or not os.path.isdir(vault_path):
        st_object.error(f"❌ Error: OBSIDIAN_VAULT_PATH environment variable is not set or is not a valid directory.")
        return

    md_files = find_markdown_files(vault_path)
    if not md_files:
        st_object.warning("⚠️ No markdown files found in the specified path.")
        return

    st_object.info(f"Found {len(md_files)} Markdown files in vault. Checking for new notes...")
    
    files_to_process = []
    skipped_count = 0

    # Identify files that need to be processed
    for file_path in md_files:
        source_name = os.path.basename(file_path)
        if collection.get(where={"source": source_name})['ids']:
            skipped_count += 1
            continue
        files_to_process.append(file_path)

    if not files_to_process:
        st_object.success(f"✅ Vault sync complete. Skipped {skipped_count} existing notes.")
        return

    all_chunks_for_processing = []
    all_metadatas_for_processing = []
    all_ids_for_processing = []
    
    st_object.info(f"Reading and chunking {len(files_to_process)} new notes from vault...")
    reading_progress_bar = st_object.progress(0, text="Reading files and chunking...")

    for i, file_path in enumerate(files_to_process):
        content, source_name = read_obsidian_note(file_path)
        chunks = chunk_text(content)
        
        for j, chunk in enumerate(chunks):
            all_chunks_for_processing.append(chunk)
            all_metadatas_for_processing.append({
                "source": source_name,
                "learned_at": datetime.now().isoformat()
            })
            all_ids_for_processing.append(f"{source_name}_{j}")
        
        current_progress = min((i + 1) / len(files_to_process), 1.0)
        reading_progress_bar.progress(int(current_progress * 100), text=f"Reading files and chunking... ({i + 1}/{len(files_to_process)})")
    
    reading_progress_bar.empty()

    if not all_chunks_for_processing:
        st_object.warning(f"No new content to learn from vault. Skipped {skipped_count} existing notes.")
        return

    # --- Start ingestion progress bar ---
    total_chunks_to_add = len(all_chunks_for_processing)
    st_object.info(f"Adding {total_chunks_to_add} total chunks from new vault notes to the knowledge base...")
    ingestion_progress_bar = st_object.progress(0, text="Ingesting chunks to ChromaDB...")

    BATCH_SIZE = 100 # Adjust batch size based on performance
    
    for i in range(0, total_chunks_to_add, BATCH_SIZE):
        batch_chunks = all_chunks_for_processing[i:i + BATCH_SIZE]
        batch_metadatas = all_metadatas_for_processing[i:i + BATCH_SIZE]
        batch_ids = all_ids_for_processing[i:i + BATCH_SIZE]

        try:
            collection.add(documents=batch_chunks, metadatas=batch_metadatas, ids=batch_ids)
        except Exception as e:
            st_object.error(f"❌ Error adding batch {i}-{i+len(batch_chunks)} from vault to ChromaDB: {e}")
            ingestion_progress_bar.empty()
            return

        current_progress = min((i + len(batch_chunks)) / total_chunks_to_add, 1.0)
        ingestion_progress_bar.progress(int(current_progress * 100), text=f"Ingesting chunks to ChromaDB... ({i + len(batch_chunks)}/{total_chunks_to_add})")
    
    ingestion_progress_bar.empty()
    st_object.success(f"✅ Vault sync complete. Learned from {len(files_to_process)} new notes. Skipped {skipped_count} existing notes.")

# Remaining functions in db_utils.py are unchanged from your last provided code.
# I'm including them here for completeness, but the changes are only in the two functions above.

def add_information(collection, info_text):
    """Adds a single piece of text information to the knowledge base."""
    source_name = f"manual_note_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    collection.add(
        documents=[info_text],
        metadatas=[{"source": source_name, "learned_at": datetime.now().isoformat()}],
        ids=[source_name]
    )
    return "✅ Manual note added successfully."

def get_reranked_context(collection, cross_encoder, query, where_filter=None, n_results=10, top_k=5):
    """Retrieves and re-ranks documents from ChromaDB to get the most accurate context."""
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        where=where_filter
    )

    initial_docs = results['documents'][0]
    if not initial_docs:
        return None

    pairs = [[query, doc] for doc in initial_docs]
    scores = cross_encoder.predict(pairs, show_progress_bar=False)

    scored_docs = sorted(zip(scores, initial_docs), reverse=True)
    return [doc for score, doc in scored_docs[:top_k]]

def manage_knowledge_list(collection):
    """Lists all unique learned sources from the knowledge base."""
    results = collection.query(query_texts=[""], n_results=1000000, include=["metadatas"]) # Query all to get all metadata
    # The previous `collection.get(include=["metadatas"])` without query_texts sometimes
    # doesn't return all documents in some ChromaDB versions/setups.
    # Using a broad query with a high n_results is a more robust way to get all.
    
    if not results['ids']:
        return "The knowledge base is empty."
        
    sources = sorted(list(set(meta['source'] for meta in results['metadatas'][0] if meta and 'source' in meta)))
    
    if not sources:
        return "No sources found in the knowledge base."
        
    return "--- Learned Sources ---\n" + "\n".join(f"  - {source}" for source in sources)


def manage_knowledge_delete(collection, source_to_delete):
    """Deletes all chunks associated with a specific source."""
    if not source_to_delete:
        return "⚠️ Please provide a source name to delete."
        
    # Use query to get IDs, as `get` with `where` might sometimes be tricky
    results = collection.query(query_texts=[""], n_results=1000000, where={"source": source_to_delete})
    ids_to_delete = results['ids'][0] if results['ids'] else []

    if not ids_to_delete:
        return f"❌ Source '{source_to_delete}' not found."
    
    collection.delete(ids=ids_to_delete)
    return f"✅ Successfully deleted {len(ids_to_delete)} chunks from source '{source_to_delete}'."


def build_knowledge_graph(collection, query):
    """Builds a NetworkX graph of connected notes."""
    # Ensure query_texts is not empty or None
    if not query:
        return nx.Graph() # Return empty graph if no query

    main_results = collection.query(query_texts=[query], n_results=1, include=["metadatas", "embeddings"]) # Get embeddings too
    
    # Check if main_results['ids'][0] is not empty before accessing
    if not main_results['ids'] or not main_results['ids'][0]:
        return nx.Graph() # Return empty graph if no results for the query
    
    central_node_id = main_results['ids'][0][0]
    central_node_source = main_results['metadatas'][0][0].get('source', 'Unknown Source')
    central_node_embedding = main_results['embeddings'][0][0] # Get the embedding of the central node

    G = nx.Graph()
    G.add_node(central_node_id, label=central_node_source, title=central_node_source, color="#FF4B4B", size=25)
    
    # Query for neighbors using the central node's embedding
    # Use the embedding directly for query_embeddings
    neighbors = collection.query(
        query_embeddings=[central_node_embedding], # Use the embedding here
        n_results=10,
        include=["metadatas"]
    )

    # Ensure neighbors['ids'][0] is not empty before iterating
    if not neighbors['ids'] or not neighbors['ids'][0]:
        return G # Return graph with just the central node if no neighbors

    for i, neighbor_id in enumerate(neighbors['ids'][0]):
        if neighbor_id == central_node_id:
            continue
        
        # Ensure metadata exists for the current neighbor
        if i < len(neighbors['metadatas'][0]):
            neighbor_metadata = neighbors['metadatas'][0][i]
            neighbor_source = neighbor_metadata.get('source', 'Unknown Source')
        else:
            neighbor_source = 'Unknown Source' # Fallback if metadata is missing for some reason

        existing_nodes = [n for n, d in G.nodes(data=True) if 'label' in d and d['label'] == neighbor_source] # Ensure 'label' key exists
        
        if existing_nodes:
            G.add_edge(central_node_id, existing_nodes[0])
        else:
            G.add_node(neighbor_id, label=neighbor_source, title=neighbor_source, color="#ADD8E6", size=15)
            G.add_edge(central_node_id, neighbor_id)
            
    return G

def get_proactive_insights(collection):
    """Finds interesting notes for the 'For You' section."""
    insights = []
    
    # 1. "On this day" feature
    today_str = datetime.now().strftime("-%m-%d")
    results = collection.get(include=["metadatas"]) # This might only return up to default limit (100)
    
    # A more robust way to get all metadata for "On this day"
    all_metadatas = []
    try:
        # Fetch all results, assuming a large enough n_results or by pagination if available
        # For simplicity, using a very large n_results; for massive DBs, proper pagination needed.
        all_results = collection.query(query_texts=[""], n_results=1000000, include=["metadatas"])
        if all_results and all_results['metadatas']:
            all_metadatas = all_results['metadatas'][0] # Access the list of dictionaries
    except Exception as e:
        print(f"Error fetching all metadatas for insights: {e}")
        # Fallback to the original less comprehensive get() if query fails
        all_metadatas = collection.get(include=["metadatas"])['metadatas']

    on_this_day_notes = []
    if all_metadatas:
        for meta in all_metadatas:
            if meta and 'learned_at' in meta and today_str in meta['learned_at']:
                on_this_day_notes.append(meta['source'])
    
    if on_this_day_notes:
        unique_notes = sorted(list(set(on_this_day_notes)))
        insights.append("**On this day, you learned about:**\n- " + "\n- ".join(unique_notes))
        
    # You could add other insights here, like "related to recent notes"
    
    return "\n\n".join(insights) if insights else "No special insights today. Time to learn something new!"