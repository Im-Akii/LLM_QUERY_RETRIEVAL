# app/services/document_loader.py

import os
import re
import requests
from urllib.parse import urlparse
from pathlib import Path

def download_file(url_or_path: str) -> bytes:
    """
    Downloads or loads a document from a URL or local path.
    Supports:
      - Azure Blob SAS URLs
      - Google Drive share links
      - Direct HTTP/HTTPS links
      - Local file paths
    Returns:
      File content as bytes
    Raises:
      Exception if file cannot be retrieved
    """

    # 1️⃣ If it's a local file path
    if os.path.exists(url_or_path):
        try:
            with open(url_or_path, "rb") as f:
                return f.read()
        except Exception as e:
            raise Exception(f"Failed to read local file: {e}")

    # 2️⃣ If it's a Google Drive share link → convert to direct download
    if "drive.google.com" in url_or_path:
        file_id_match = re.search(r"/d/([a-zA-Z0-9_-]+)", url_or_path)
        if file_id_match:
            file_id = file_id_match.group(1)
            url_or_path = f"https://drive.google.com/uc?export=download&id={file_id}"

    # 3️⃣ Otherwise, assume it's HTTP/HTTPS
    try:
        r = requests.get(url_or_path, stream=True, timeout=30)
        if r.status_code != 200:
            raise Exception(f"HTTP {r.status_code} - {r.text[:200]}")
        return r.content
    except requests.exceptions.RequestException as e:
        raise Exception(f"Network error: {e}")

    # If nothing worked
    raise Exception("Could not retrieve document")

def load_document(url_or_path: str) -> bytes:
    """
    Loads a document from the given path/URL.
    Returns:
      Document content in bytes
    """
    try:
        content = download_file(url_or_path)
        if not content:
            raise Exception("Document is empty")
        return content
    except Exception as e:
        raise Exception(f"Failed to load document from '{url_or_path}': {e}")
