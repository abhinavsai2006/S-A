"""
Igris AI Agent — File Manager Skill  (Issue #1 — OpenClaw skill)

Justification:
A capable terminal agent should be able to interact with the local filesystem —
read files, write files, list directories, and get file info. This is a foundational
skill that other skills (document reader, cloud upload) build upon.
"""

import os
from langchain.tools import tool


@tool
def read_file(file_path: str) -> str:
    """Read the contents of a text file and return them.
    Use this when the user asks to view or read a file."""
    try:
        file_path = os.path.abspath(file_path)
        if not os.path.exists(file_path):
            return f"File not found: {file_path}"
        if not os.path.isfile(file_path):
            return f"Path is not a file: {file_path}"

        size = os.path.getsize(file_path)
        if size > 1_000_000:  # 1 MB limit for safety
            return f"File is too large ({size:,} bytes). Use document_reader for large files."

        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        return content
    except Exception as e:
        return f"Error reading file: {e}"


@tool
def write_file(file_path: str, content: str) -> str:
    """Write content to a file. Creates the file if it doesn't exist.
    Use this when the user asks to create or write to a file."""
    try:
        file_path = os.path.abspath(file_path)
        dir_name = os.path.dirname(file_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Successfully wrote {len(content)} characters to {file_path}"
    except Exception as e:
        return f"Error writing file: {e}"


@tool
def list_directory(directory_path: str) -> str:
    """List files and subdirectories in a given directory.
    Use this when the user wants to see what files exist in a folder."""
    try:
        directory_path = os.path.abspath(directory_path)
        if not os.path.exists(directory_path):
            return f"Directory not found: {directory_path}"
        if not os.path.isdir(directory_path):
            return f"Path is not a directory: {directory_path}"

        items = os.listdir(directory_path)
        if not items:
            return f"Directory is empty: {directory_path}"

        result_lines = [f"Contents of {directory_path}:\n"]
        for item in sorted(items):
            full_path = os.path.join(directory_path, item)
            if os.path.isdir(full_path):
                result_lines.append(f"  📁 {item}/")
            else:
                size = os.path.getsize(full_path)
                result_lines.append(f"  📄 {item}  ({size:,} bytes)")

        return "\n".join(result_lines)
    except Exception as e:
        return f"Error listing directory: {e}"


@tool
def get_file_info(file_path: str) -> str:
    """Get detailed information about a file (size, modified date, type).
    Use this when the user asks about a specific file's details."""
    try:
        file_path = os.path.abspath(file_path)
        if not os.path.exists(file_path):
            return f"Path not found: {file_path}"

        stat = os.stat(file_path)
        import datetime
        modified = datetime.datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        created = datetime.datetime.fromtimestamp(stat.st_ctime).strftime("%Y-%m-%d %H:%M:%S")
        is_dir = os.path.isdir(file_path)

        return (
            f"Path: {file_path}\n"
            f"Type: {'Directory' if is_dir else 'File'}\n"
            f"Size: {stat.st_size:,} bytes\n"
            f"Modified: {modified}\n"
            f"Created: {created}"
        )
    except Exception as e:
        return f"Error getting file info: {e}"


FILE_MANAGER_TOOLS = [read_file, write_file, list_directory, get_file_info]
