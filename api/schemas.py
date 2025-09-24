from pydantic import BaseModel, Field
from typing import List, Dict, Optional

class NoteInput(BaseModel):
    note_name: str = Field(..., description="The name of the new note.")
    document_paths: List[str] = Field(..., description="List of paths to the documents to be processed.")

class MindMapResponse(BaseModel):
    mindmap_json: Dict = Field(..., description="The generated mind map in JSON format.")

class BriefReportResponse(BaseModel):
    outline: Dict = Field(..., description="The generated outline for the brief report.")
    report: str = Field(..., description="The generated brief report in Markdown format.")

class CustomReportStartRequest(BaseModel):
    note_id: str = Field(..., description="The ID of the note to generate the report from.")
    title: str = Field(..., description="The title of the custom report.")
    requirements: str = Field(..., description="The user's requirements for the report.")

class UserFeedbackRequest(BaseModel):
    session_id: str = Field(..., description="The WebSocket session ID for the ongoing report generation.")
    feedback_type: str = Field(..., description="Type of feedback, e.g., 'start_generation', 'approve_outline', 'revise_outline'.")
    feedback_content: Optional[str] = Field(None, description="User's feedback or revision comments.")
    
    # Fields for starting the generation
    note_id: Optional[str] = Field(None, description="The ID of the note to generate the report from.")
    title: Optional[str] = Field(None, description="The title of the custom report.")
    requirements: Optional[str] = Field(None, description="The user's requirements for the report.")
