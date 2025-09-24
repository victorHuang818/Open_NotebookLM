from typing import TypedDict, Annotated, List, Dict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt

# Temporarily add youtu-graphrag to python path for imports
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from notebookLM.tools.retrieval_tools import RetrievalTools
from utils.call_llm_api import LLMCompletionCall


# 1. Define the state for our report generation agent
class ReportAgentState(TypedDict):
    # `messages` is a required field in LangGraph.
    # It will store the history of interactions.
    # `add_messages` is a special operator that appends new messages
    # to the list, rather than overwriting it.
    messages: Annotated[list, add_messages]
    
    # The note ID tells us which knowledge graph to use.
    note_id: str
    
    # User's initial request for the report
    report_title: str
    user_requirements: str

    # The generated outline for user review
    outline: Dict

    # The final generated report content
    report_content: str
    
    # User's feedback for revisions
    user_feedback: str
    
    # A field to track the current step of the process
    current_step: str


class ReportAgent:
    """
    An agent that orchestrates the generation of custom reports with user interaction.
    """
    def __init__(self):
        self.llm_client = LLMCompletionCall()
        
        # Build the state machine (graph)
        self.graph = self._build_graph()

    def _build_graph(self):
        """
        Builds the LangGraph state machine for the report generation process.
        """
        builder = StateGraph(ReportAgentState)

        # Initialize tools once for the agent
        # Note: In a multi-note scenario, tool instances should be managed per session.
        # For simplicity, we create them here but they are note-specific.
        tools = RetrievalTools(note_id="demo") # Placeholder, will be dynamic in a real app
        tool_node = ToolNode([
            tools.get_knowledge_overview, 
            tools.search_topic_details, 
            tools.find_connections_between_topics
        ])

        # Define the nodes of the graph
        builder.add_node("start_generation", self.start_generation_node)
        builder.add_node("generate_outline", self.generate_outline_node)
        builder.add_node("await_outline_feedback", self.await_feedback_node)
        builder.add_node("write_report", self.write_report_node)
        builder.add_node("tools", tool_node) # Add the tool node

        # Define the edges
        builder.add_edge(START, "start_generation")
        builder.add_edge("start_generation", "generate_outline")
        
        # After generating the outline, wait for feedback
        builder.add_edge("generate_outline", "await_outline_feedback")

        # After getting feedback, decide where to go next
        builder.add_conditional_edges(
            "await_outline_feedback",
            self.should_continue_or_revise,
            {
                "revise": "generate_outline", # If feedback is for revision, go back
                "continue": "write_report"    # If approved, proceed to write report
            }
        )
        
        # After writing the report, the process ends
        builder.add_edge("write_report", END)
        
        from langgraph.checkpoint.memory import InMemorySaver
        checkpointer = InMemorySaver()
        
        return builder.compile(checkpointer=checkpointer)

    # --- Router & Conditional Logic ---

    def should_continue_or_revise(self, state: ReportAgentState) -> str:
        """
        Determines the next step based on user feedback.
        """
        if state.get("user_feedback") and "revise" in state["user_feedback"].lower():
            print("--- [Agent] User requested revisions. Regenerating outline. ---")
            return "revise"
        else:
            print("--- [Agent] Outline approved. Proceeding to write report. ---")
            return "continue"

    # --- Node Implementations ---

    def start_generation_node(self, state: ReportAgentState) -> Dict:
        """
        Initializes the process based on the user's starting request.
        """
        return {"current_step": "generating_outline"}

    def generate_outline_node(self, state: ReportAgentState) -> Dict:
        """
        Generates the initial report outline using the retrieval tools.
        """
        note_id = state["note_id"]
        title = state["report_title"]
        requirements = state["user_requirements"]
        feedback = state.get("user_feedback", "No feedback provided.") # Use feedback if available
        
        print(f"--- [Agent] Generating outline for '{title}' using note '{note_id}' ---")
        if "revise" in feedback.lower():
            print(f"--- [Agent] Incorporating feedback: {feedback} ---")

        # Initialize the tools for the specific note
        tools = RetrievalTools(note_id=note_id)

        # Here, a more sophisticated agent would decompose requirements and call tools for each part.
        # For this first version, we'll use the knowledge overview as the basis for the outline.
        knowledge_overview = tools.get_knowledge_overview()
        
        prompt = f"""
        Based on the user's request, available knowledge, and the following feedback, generate or revise a structured outline for a report.

        **Report Title:** {title}
        **User Requirements:** {requirements}
        **User Feedback for Revision:** {feedback}

        **Available Knowledge Overview (from document communities):**
        {knowledge_overview}

        **Task:**
        Create a report outline in a JSON format. If feedback is provided, prioritize addressing the feedback.

        **JSON Output Format:**
        {{
            "title": "The Report Title",
            "chapters": [
                {{"title": "Chapter 1 Title", "summary": "A brief summary of what this chapter will cover."}},
                {{"title": "Chapter 2 Title", "summary": "..."}}
            ]
        }}
        """
        
        response = self.llm_client.call_api(prompt)
        import json_repair
        outline = json_repair.loads(response)
        
        # Clear feedback after using it
        return {"outline": outline, "current_step": "awaiting_outline_feedback", "user_feedback": None}

    def await_feedback_node(self, state: ReportAgentState) -> None:
        """
        Pauses the graph execution to wait for human-in-the-loop feedback.
        """
        print("--- [Agent] Awaiting user feedback on the generated outline... ---")
        return interrupt()

    def write_report_node(self, state: ReportAgentState) -> Dict:
        """
        Writes the full report content based on the approved outline.
        """
        note_id = state["note_id"]
        outline = state["outline"]
        title = state["report_title"]
        
        print(f"--- [Agent] Writing full report for '{title}' ---")
        tools = RetrievalTools(note_id=note_id)
        
        report_chapters = []
        for chapter in outline.get("chapters", []):
            chapter_title = chapter.get("title")
            print(f"--- [Agent] Writing chapter: {chapter_title} ---")
            
            # Use a tool to get details for this chapter title
            chapter_context = tools.search_topic_details(topic_query=chapter_title, search_depth="summary")
            
            prompt = f"""
            You are an expert report writer. Write a detailed chapter for a report.

            **Report Title:** {title}
            **Chapter Title:** {chapter_title}

            **Retrieved Context for this Chapter:**
            {chapter_context}

            **Task:**
            Write the full content for this chapter in Markdown format. Ensure the content is detailed, well-structured, and based on the provided context.
            """
            
            chapter_content = self.llm_client.call_api(prompt)
            report_chapters.append(f"## {chapter_title}\n\n{chapter_content}")
            
        full_report = f"# {title}\n\n" + "\n\n".join(report_chapters)
        
        return {"report_content": full_report, "current_step": "completed"}

if __name__ == "__main__":
    # This is an example of how to run the agent.
    
    agent = ReportAgent()
    graph = agent.graph

    # Let's simulate the start of a custom report request
    initial_input = {
        "messages": [], # Start with an empty message list
        "note_id": "demo",
        "report_title": "Analysis of Knowledge Systems",
        "user_requirements": "Provide a report on community detection and its relation to knowledge graphs."
    }
    
    # We use a config with a thread_id to ensure the state is managed for this specific session
    config = {"configurable": {"thread_id": "report-session-1"}}
    
    print("--- [User] Starting report generation ---")
    
    # Stream the graph execution
    for chunk in graph.stream(initial_input, config=config):
        # The output of each node is printed as it executes
        print("\n--- [Graph Step Output] ---")
        print(chunk)
        print("--------------------------")

    # At this point, the graph is interrupted and waiting.
    # We can check the state.
    final_state = graph.get_state(config)
    print("\n--- [Agent Paused] ---")
    print(f"Current step: {final_state.values['current_step']}")
    print("Generated Outline:")
    import json
    print(json.dumps(final_state.values['outline'], indent=2))
    print("----------------------")

    # --- SIMULATE USER INTERACTION ---
    
    # Scenario 1: User approves the outline
    print("\n--- [User] Approving outline ---")
    
    # We continue the execution by streaming from the last known state (the interruption point)
    # We don't provide any feedback, so the conditional edge will go to "continue"
    for chunk in graph.stream(None, config=config):
        print("\n--- [Graph Step Output] ---")
        print(chunk)
        print("--------------------------")

    final_state_approved = graph.get_state(config)
    print("\n--- [Agent Finished] ---")
    print(f"Current step: {final_state_approved.values['current_step']}")
    print("\nGenerated Report (first 400 chars):")
    print(final_state_approved.values['report_content'][:400] + "...")
    print("----------------------")

    # Scenario 2: User requests a revision (requires running from the start with a new session)
    print("\n\n" + "="*50)
    print("--- [User] Starting a new session to test revision ---")
    config_revision = {"configurable": {"thread_id": "report-session-2"}}
    
    # Run until the first interruption
    for chunk in graph.stream(initial_input, config=config_revision):
        pass # We already saw these outputs

    print("\n--- [Agent Paused] ---")
    print("Generated Outline for Revision:")
    first_outline = graph.get_state(config_revision).values['outline']
    import json
    print(json.dumps(first_outline, indent=2))
    print("----------------------")

    print("\n--- [User] Requesting revision ---")
    # To continue after an interruption, we pass the input for the *next* node in the graph.
    # In our case, the interruption happens AT `await_feedback_node`. The next input will be processed by this node
    # and then passed to the conditional edge. However, LangGraph's `update_state` is a better way.
    
    # Let's update the state with the user's feedback
    graph.update_state(config_revision, {"user_feedback": "Please revise the outline to include a chapter on 'Future Trends'."})

    # And now, continue the stream
    for chunk in graph.stream(None, config=config_revision):
        print("\n--- [Graph Step Output] ---")
        print(chunk)
        print("--------------------------")
        
    final_state_revised = graph.get_state(config_revision)
    print("\n--- [Agent Paused After Revision] ---")
    print("Revised Outline:")
    print(json.dumps(final_state_revised.values['outline'], indent=2))
    print("----------------------")
