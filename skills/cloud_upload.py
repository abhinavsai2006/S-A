"""
Igris AI Agent — Cloud Upload Skill  (Issue #3 — cloud upload)

Provides Google Drive upload capability.
Uses the Google Drive API v3 via google-api-python-client.

Setup required:
1. Create a Google Cloud project
2. Enable the Google Drive API
3. Create OAuth2 credentials (desktop app)
4. Save credentials JSON to igris_data/gdrive_credentials.json
5. On first run, a browser window will open for authorization
"""

import os
from langchain.tools import tool
from config import GOOGLE_DRIVE_CREDENTIALS_FILE, GOOGLE_DRIVE_UPLOAD_FOLDER_ID


def _get_drive_service():
    """Authenticate and return a Google Drive API service object."""
    try:
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from google.auth.transport.requests import Request
        from googleapiclient.discovery import build
    except ImportError:
        return None, ("Google API libraries not installed. Run:\n"
                      "pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib")

    SCOPES = ["https://www.googleapis.com/auth/drive.file"]
    token_path = os.path.join(os.path.dirname(GOOGLE_DRIVE_CREDENTIALS_FILE), "gdrive_token.json")

    creds = None
    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists(GOOGLE_DRIVE_CREDENTIALS_FILE):
                return None, (f"Credentials file not found at {GOOGLE_DRIVE_CREDENTIALS_FILE}.\n"
                              "Please set up Google Drive API credentials first.")
            flow = InstalledAppFlow.from_client_secrets_file(GOOGLE_DRIVE_CREDENTIALS_FILE, SCOPES)
            creds = flow.run_local_server(port=0)

        with open(token_path, "w") as token_file:
            token_file.write(creds.to_json())

    service = build("drive", "v3", credentials=creds)
    return service, None


@tool
def upload_to_drive(file_path: str) -> str:
    """Upload a file to Google Drive.
    Use this when the user wants to upload a file to the cloud.
    Requires Google Drive API credentials to be set up first."""
    try:
        file_path = os.path.abspath(file_path)
        if not os.path.exists(file_path):
            return f"File not found: {file_path}"
        if not os.path.isfile(file_path):
            return f"Path is not a file: {file_path}"

        service, error = _get_drive_service()
        if error:
            return error

        from googleapiclient.http import MediaFileUpload

        file_name = os.path.basename(file_path)
        file_metadata = {"name": file_name}

        if GOOGLE_DRIVE_UPLOAD_FOLDER_ID:
            file_metadata["parents"] = [GOOGLE_DRIVE_UPLOAD_FOLDER_ID]

        media = MediaFileUpload(file_path, resumable=True)
        result = service.files().create(
            body=file_metadata, media_body=media, fields="id,name,webViewLink"
        ).execute()

        return (
            f"✅ File uploaded successfully!\n"
            f"Name: {result.get('name')}\n"
            f"ID: {result.get('id')}\n"
            f"Link: {result.get('webViewLink', 'N/A')}"
        )
    except Exception as e:
        return f"Upload failed: {e}"


@tool
def list_drive_files() -> str:
    """List recent files in Google Drive.
    Use this when the user wants to see their uploaded files."""
    try:
        service, error = _get_drive_service()
        if error:
            return error

        results = service.files().list(
            pageSize=10,
            fields="files(id, name, mimeType, modifiedTime, webViewLink)",
            orderBy="modifiedTime desc"
        ).execute()

        files = results.get("files", [])
        if not files:
            return "No files found in Google Drive."

        lines = ["📁 Recent Google Drive files:\n"]
        for f in files:
            name = f.get("name", "Unknown")
            modified = f.get("modifiedTime", "Unknown")
            link = f.get("webViewLink", "N/A")
            lines.append(f"  • {name}  (modified: {modified})\n    Link: {link}")

        return "\n".join(lines)
    except Exception as e:
        return f"Error listing Drive files: {e}"


CLOUD_UPLOAD_TOOLS = [upload_to_drive, list_drive_files]
