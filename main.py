import fastapi
from fastapi import FastAPI, HTTPException, WebSocket
import uvicorn
from typing import Dict

# Temporarily add youtu-graphrag to python path for imports
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from notebookLM.api import schemas
from models.constructor import kt_gen
from config import get_config
from notebookLM.api import endpoints as notebooklm_endpoints
from notebookLM.agents.report_agent import ReportAgent
from utils.call_llm_api import LLMCompletionCall
import json
import json_repair

app = FastAPI(
    title="notebookLM API",
    description="API for creating notes, generating mind maps, and authoring reports.",
    version="1.0.0",
)

# In-memory storage for active WebSocket connections and their associated agent sessions
# In a production environment, you would replace this with a more robust solution like Redis
active_connections: Dict[str, WebSocket] = {}

# In-memory storage for active agent sessions, keyed by a unique session ID (e.g., client's connection ID)
agent_sessions: Dict[str, ReportAgent] = {}


# --- Helper Functions ---

def _generate_and_save_preliminary_schema(note_name: str, doc_paths: list) -> str:
    """
    Generates a preliminary schema based on note and document titles using an LLM.
    """
    print(f"--- Generating preliminary schema for note: {note_name} ---")
    llm_client = LLMCompletionCall()
    
    # Extract titles from file paths
    doc_titles = [os.path.basename(path).split('.')[0] for path in doc_paths]
    
    context = f"Note Title: {note_name}\nDocument Titles: {', '.join(doc_titles)}"

    prompt = f"""
    You are an expert data architect. Based on the following note and document titles, please generate a preliminary knowledge graph schema.
    The schema should define potential entity types (Nodes), relationship types (Relations), and attribute types (Attributes).

    **Context:**
    {context}

    **Task:**
    Generate a JSON object containing three keys: "Nodes", "Relations", and "Attributes".
    - "Nodes": A list of strings representing potential entity types (e.g., "Person", "Company", "Project").
    - "Relations": A list of strings for potential relationships (e.g., "works_for", "invested_in", "related_to").
    - "Attributes": A list of strings for potential properties of the nodes (e.g., "role", "location", "status").

    **Example Output:**
    {{
      "Nodes": ["Person", "Organization", "Technology"],
      "Relations": ["develops", "manages", "competes_with"],
      "Attributes": ["position", "headquarters", "release_date"]
    }}

    Please provide ONLY the JSON object as your response.
    """
    
    try:
        response = llm_client.call_api(prompt)
        schema_json = json_repair.loads(response)

        # Validate basic structure
        if not all(k in schema_json for k in ["Nodes", "Relations", "Attributes"]):
            raise ValueError("LLM response did not contain the required schema keys.")

        # Ensure schemas directory exists
        schema_dir = "schemas"
        os.makedirs(schema_dir, exist_ok=True)
        
        schema_path = os.path.join(schema_dir, f"{note_name}.json")
        with open(schema_path, 'w', encoding='utf-8') as f:
            json.dump(schema_json, f, ensure_ascii=False, indent=2)
            
        print(f"--- Preliminary schema saved to {schema_path} ---")
        return schema_path
    except Exception as e:
        print(f"Failed to generate or save preliminary schema: {e}")
        # Fallback to an empty schema if generation fails
        schema_dir = "schemas"
        os.makedirs(schema_dir, exist_ok=True)
        schema_path = os.path.join(schema_dir, f"{note_name}.json")
        with open(schema_path, 'w', encoding='utf-8') as f:
            json.dump({ "Nodes": [], "Relations": [], "Attributes": [] }, f)
        return schema_path


def run_graph_construction(note_name: str, doc_paths: list):
    """Initiates and runs the knowledge graph construction process for a new note."""
    try:
        config = get_config()

        # Generate the preliminary schema before initializing the builder
        preliminary_schema_path = _generate_and_save_preliminary_schema(note_name, doc_paths)

        # The KTBuilder uses the dataset_name (which we use as note_name)
        # to find the correct paths in the config. We pass the generated schema path directly.
        builder = kt_gen.KTBuilder(
            dataset_name=note_name, 
            config=config, 
            schema_path=preliminary_schema_path,
            mode='agent' # Ensure agent mode is active for schema evolution
        )
        
        # We need to load documents from the provided paths.
        # This part assumes documents are in a format that KTBuilder can process.
        # For youtu-graphrag, it expects a list of dicts from a JSON file.
        # Here we'll simulate that by assuming doc_paths point to JSON files.
        all_docs = []
        for path in doc_paths:
            if os.path.exists(path):
                # Assuming kt_gen's load format
                with open(path, 'r', encoding='utf-8') as f:
                    # In youtu-graphrag, it loads a list of dicts directly
                    # We will assume a similar structure.
                    import json
                    try:
                        docs = json.load(f)
                        if isinstance(docs, list):
                            all_docs.extend(docs)
                        else:
                            # Handle cases where the JSON is not a list of documents
                            all_docs.append(docs)
                    except json.JSONDecodeError:
                        # Handle plain text files if needed
                        f.seek(0)
                        all_docs.append({"text": f.read()})

            else:
                raise FileNotFoundError(f"Document not found at path: {path}")

        if not all_docs:
            raise ValueError("No documents found or loaded from the provided paths.")

        builder.process_all_documents(all_docs)
        # Save the graph
        output_path = config.get_dataset_config(note_name).graph_output
        builder.save_graphml(output_path.replace(".json", ".graphml"))
        
        # The JSON output is handled inside build_knowledge_graph, but we can call it manually
        # by formatting the output. Let's ensure the JSON is saved.
        json_output_path = output_path
        os.makedirs(os.path.dirname(json_output_path), exist_ok=True)
        output_data = builder.format_output()
        with open(json_output_path, 'w', encoding='utf-8') as f:
            import json
            json.dump(output_data, f, ensure_ascii=False, indent=2)

    except Exception as e:
        # Log the exception for debugging
        print(f"Error during graph construction: {e}")
        raise

# --- API Endpoints ---

@app.post("/notes", status_code=201)
async def create_note(note_input: schemas.NoteInput):
    """
    Creates a new note by processing the provided documents and building
    a knowledge graph.
    """
    try:
        # Here, we need to ensure the config file has an entry for the new note_name
        # For this example, we'll assume it can be handled dynamically or
        # that a template entry exists.
        # For a real application, you'd likely need to manage this more robustly.
        print(f"Starting knowledge graph construction for note: {note_input.note_name}")
        run_graph_construction(note_input.note_name, note_input.document_paths)
        print(f"Successfully created knowledge graph for note: {note_input.note_name}")
        return {"message": f"Note '{note_input.note_name}' created and processed successfully."}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create note: {str(e)}")


# --- WebSocket Endpoint for Custom Reports ---

@app.websocket("/ws/report/custom/{session_id}")
async def custom_report_websocket(websocket: WebSocket, session_id: str):
    """
    Handles the interactive, stateful process of generating a custom report.
    """
    await websocket.accept()
    active_connections[session_id] = websocket
    
    # Each connection gets its own agent instance to manage its state
    agent = ReportAgent()
    agent_sessions[session_id] = agent
    graph = agent.graph
    
    config = {"configurable": {"thread_id": session_id}}

    try:
        while True:
            # Wait for a message from the client
            request_json = await websocket.receive_json()
            request = schemas.UserFeedbackRequest(**request_json) # Using a generic model for now

            if request.feedback_type == "start_generation":
                initial_input = {
                    "messages": [],
                    "note_id": request.note_id,
                    "report_title": request.title,
                    "user_requirements": request.requirements
                }
                
                # Stream the graph execution until the first interruption
                async for chunk in graph.astream(initial_input, config=config):
                    step_name = list(chunk.keys())[0]
                    step_output = chunk[step_name]
                    
                    # Send status updates to the client
                    await websocket.send_json({
                        "type": "status_update",
                        "payload": {"message": f"Step '{step_name}' completed."}
                    })

                # Once interrupted, get the state and send the outline for review
                final_state = graph.get_state(config)
                await websocket.send_json({
                    "type": "outline_ready_for_review",
                    "payload": {"outline": final_state.values['outline']}
                })
            
            elif request.feedback_type in ["approve_outline", "revise_outline"]:
                feedback = request.feedback_content or "User approved."
                if request.feedback_type == "revise_outline":
                    feedback = f"revise: {feedback}"

                # Update the state with the user's feedback
                graph.update_state(config, {"user_feedback": feedback})
                
                # Continue streaming the graph from where it left off
                async for chunk in graph.astream(None, config=config):
                    step_name = list(chunk.keys())[0]
                    step_output = chunk[step_name]
                    await websocket.send_json({
                        "type": "status_update",
                        "payload": {"message": f"Step '{step_name}' completed."}
                    })
                
                final_state = graph.get_state(config)
                if final_state.values.get("current_step") == "awaiting_outline_feedback":
                     await websocket.send_json({
                        "type": "outline_ready_for_review",
                        "payload": {"outline": final_state.values['outline']}
                    })
                elif final_state.values.get("current_step") == "completed":
                    await websocket.send_json({
                        "type": "report_ready_for_review",
                        "payload": {"report_content": final_state.values['report_content']}
                    })

    except WebSocketDisconnect:
        print(f"Client {session_id} disconnected.")
        if session_id in active_connections:
            del active_connections[session_id]
        if session_id in agent_sessions:
            del agent_sessions[session_id]
    except Exception as e:
        print(f"An error occurred in WebSocket for session {session_id}: {e}")
        await websocket.send_json({"type": "error", "payload": {"message": str(e)}})


app.include_router(notebooklm_endpoints.router, prefix="/notebooklm", tags=["NotebookLM"])


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
