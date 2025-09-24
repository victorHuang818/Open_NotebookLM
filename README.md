# notebookLM Application Module

This directory contains the core backend logic for the "notebookLM" application, a powerful tool for knowledge synthesis, mind mapping, and intelligent report generation.

## Features

1.  **Dynamic Note Creation**: Users can create a "note" by providing a title and one or more source documents. The system processes these documents to build a dedicated, hierarchical knowledge graph, forming the foundation for all subsequent features.

2.  **Automated Mind Map Generation**: Creates a structured, hierarchical mind map from the knowledge graph, providing a clear visual overview of the core concepts and their interconnections within the documents.

3.  **Brief Report Generation**: A non-interactive feature that automatically generates a concise, well-structured report. It first creates an outline based on the main themes (communities) identified in the knowledge graph and then writes the full report content.

4.  **Custom Report Generation (Interactive Agent)**: A sophisticated, stateful feature powered by a `LangGraph`-based ReAct Agent. This allows for a conversational and iterative report-writing process:
    *   **User-Driven**: Users provide a title and specific requirements.
    *   **Outline Approval**: The agent generates a draft outline for the user to review.
    *   **Iterative Feedback**: Users can approve the outline or provide feedback for revisions. The agent will regenerate the outline until it meets the user's needs.
    *   **Final Report Generation**: Once the outline is approved, the agent proceeds to write the full report, which can also be reviewed and revised.

## Technical Architecture & Advantages

1.  **Powered by `youtu-graphrag`**: The application leverages the robust `youtu-graphrag` engine as its core knowledge processing layer. This provides:
    *   **Hierarchical Knowledge Graphs**: A multi-layered graph structure (attributes, relations, keywords, communities) that captures both fine-grained details and high-level themes.
    *   **Advanced Retrieval**: Utilizes a powerful, multi-path retrieval system (`KTRetriever` and `DualFAISSRetriever`) for efficient and accurate information lookup.
    *   **Dynamic Schema Evolution**: The system can intelligently adapt and expand its understanding of the content by evolving the knowledge schema during processing.

2.  **Stateful Agent for Complex Tasks**: For the custom report feature, we use `LangGraph` to build a stateful agent. This is a significant advantage over simple LLM chains because it allows for:
    *   **Human-in-the-Loop**: The process can pause and wait for user input, making the generation process truly interactive and controllable.
    *   **Reliable Multi-Step Logic**: The graph-based structure ensures that complex, multi-step tasks with conditional logic (like revisions) are executed reliably.
    *   **Modularity**: The agent's logic is cleanly separated from the tools it uses.

3.  **Modular and Extensible Design**: The `notebookLM` module is designed as a self-contained application layer that sits on top of the `youtu-graphrag` engine.
    *   **Clear Separation of Concerns**: The API, services, agents, and tools are organized into distinct modules with clear responsibilities.
    *   **Abstracted Tool Layer**: A dedicated tool layer acts as an adapter between the agent and the underlying retrieval engine, making it easy to add new capabilities or modify existing ones without changing the agent's core logic.

4.  **Modern API Design**:
    *   **Hybrid API Strategy**: The application effectively uses the right protocol for the right job:
        *   **REST API**: For simple, stateless requests like generating a mind map or a brief report.
        *   **WebSocket**: For the complex, stateful, and real-time communication required by the custom report agent.
    *   **Pydantic Schemas**: All API inputs and outputs are strongly typed using Pydantic, ensuring data validation and clear API contracts.

## How to Run

1.  **Ensure `youtu-graphrag` is set up**: Make sure all dependencies from the root `requirements.txt` are installed.
2.  **Build a Knowledge Graph**: Before using the API, you must first create a note by sending a POST request to the `/notes` endpoint. This will build the necessary graph and indices. (Note: Ensure your `config/base_config.yaml` is configured to handle the `note_name` you provide).
3.  **Run the FastAPI Server**:
    ```bash
    python notebookLM/main.py
    ```
4.  **Interact with the API**:
    *   Access the auto-generated documentation at `http://localhost:8000/docs`.
    *   Use a client to interact with the REST endpoints (`/notebooklm/mindmap`, `/notebooklm/brief-report`).
    *   Use a WebSocket client to connect to `/ws/report/custom/{session_id}` for custom report generation.

## Acknowledgements

The core knowledge graph construction and retrieval engine of this application is built upon the powerful open-source project [youtu-graphrag](https://github.com/TencentCloudADP/youtu-graphrag). We extend our sincere gratitude to the original authors for their outstanding work and for making it available to the community.
