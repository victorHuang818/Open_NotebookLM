from fastapi import APIRouter, HTTPException
from notebookLM.services.mindmap_service import MindMapService
from notebookLM.services.report_service import ReportService
from notebookLM.api import schemas

router = APIRouter()

mindmap_service = MindMapService()
report_service = ReportService()

@router.post("/mindmap", response_model=schemas.MindMapResponse)
async def generate_mindmap_endpoint(note_id: str, note_title: str):
    """
    Generates a mind map for a given note ID and title.
    """
    try:
        mindmap_json = mindmap_service.generate_mindmap(note_id=note_id, note_title=note_title)
        return schemas.MindMapResponse(mindmap_json=mindmap_json)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"Knowledge graph for note '{note_id}' not found. Please create the note first.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate mind map: {str(e)}")

@router.post("/brief-report", response_model=schemas.BriefReportResponse)
async def generate_brief_report_endpoint(note_id: str, note_title: str):
    """
    Generates a brief, non-interactive report for a given note ID and title.
    """
    try:
        outline, report = report_service.generate_brief_report(note_id=note_id, note_title=note_title)
        return schemas.BriefReportResponse(outline=outline, report=report)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"Knowledge graph for note '{note_id}' not found. Please create the note first.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate brief report: {str(e)}")
