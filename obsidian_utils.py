import os
import frontmatter

def find_markdown_files(vault_path):
    """Finds all markdown files in the specified vault path."""
    md_files = []
    for root, _, files in os.walk(vault_path):
        for file in files:
            if file.endswith(".md"):
                md_files.append(os.path.join(root, file))
    return md_files

def read_obsidian_note(file_path):
    """Reads a single Obsidian note, parsing its content and metadata (like tags)."""
    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            note = frontmatter.load(f)
            content = note.content
            # Convert Obsidian tags from metadata into a queryable string in the content
            tags = note.metadata.get('tags', [])
            if tags:
                # Ensure tags are a list, as they can sometimes be a single string
                if isinstance(tags, str):
                    tags = [tags]
                tag_string = " ".join([f"#{tag}" for tag in tags])
                content = f"Tags: {tag_string}\n\n{content}"
            return content, os.path.basename(file_path)
        except Exception as e:
            # If frontmatter fails, just read the raw content
            print(f"Could not parse frontmatter for {file_path}: {e}. Reading raw content.")
            f.seek(0)
            return f.read(), os.path.basename(file_path)