import json
from typing import Dict, List, Tuple

# Temporarily add youtu-graphrag to python path for imports
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from notebookLM.tools.retrieval_tools import RetrievalTools
from utils.call_llm_api import LLMCompletionCall
import json_repair

class ReportService:
    """
    Service responsible for generating reports from a note's knowledge graph.
    Supports both brief (direct) and custom (agent-based) report generation.
    """
    def __init__(self):
        self.llm_client = LLMCompletionCall()

    def generate_brief_report(self, note_id: str, note_title: str) -> Tuple[Dict, str]:
        """
        Generates a brief, non-interactive report for a given note.

        This process involves two main steps:
        1. Generate a structured outline based on the knowledge overview.
        2. Generate the full report content based on the outline and knowledge overview.

        Args:
            note_id: The ID of the note.
            note_title: The title for the report.

        Returns:
            A tuple containing the generated outline (Dict) and the full report (str).
        """
        try:
            # 1. Get the high-level knowledge overview
            retrieval_tools = RetrievalTools(note_id=note_id)
            communities = retrieval_tools.get_knowledge_overview()

            if not communities:
                raise ValueError("No communities found to generate a report.")

            # 2. Step 1: Generate the outline
            outline_prompt = self._build_outline_prompt(note_title, communities)
            outline_response = self.llm_client.call_api(outline_prompt)
            outline = json_repair.loads(outline_response)

            # 3. Step 2: Generate the report from the outline and knowledge
            report_prompt = self._build_brief_report_prompt(note_title, outline, communities)
            report_content = self.llm_client.call_api(report_prompt)

            return outline, report_content

        except Exception as e:
            print(f"Error generating brief report for note '{note_id}': {e}")
            # Return a structured error
            error_outline = {"title": note_title, "error": str(e)}
            error_report = f"# {note_title}\n\nAn error occurred while generating the report: {e}"
            return error_outline, error_report

    def _build_outline_prompt(self, title: str, communities: List[Dict]) -> str:
        """Builds the prompt to generate an outline for the brief report."""
        community_details = self._format_community_details(communities)
        return f"""
        You are an expert report architect. Based on the following knowledge communities, create a concise and logical report outline in JSON format.

        **Report Title:** {title}

        **Available Knowledge Overview:**
        {community_details}

        **Task:**
        Generate a JSON outline with a title and a list of chapters. Each chapter should have a title and a brief summary of its content.

        **JSON Output Format:**
        {{
            "title": "The Report Title",
            "chapters": [
                {{"title": "Chapter 1 Title", "summary": "A brief summary of what this chapter will cover."}}
            ]
        }}
        """

    def _build_brief_report_prompt(self, title: str, outline: Dict, communities: List[Dict]) -> str:
        """Builds the prompt to generate the full brief report content."""
        community_details = self._format_community_details(communities)
        outline_str = json.dumps(outline, indent=2)

        return f"""
        You are a professional report writer. Your task is to write a comprehensive report based on the provided outline and knowledge overview.

        **Report Title:** {title}

        **Report Outline:**
        {outline_str}

        **Available Knowledge Overview:**
        {community_details}

        **Task:**
        Write the full report in Markdown format. Follow the structure of the provided outline. For each chapter, synthesize the information from the relevant knowledge communities to create detailed and informative content.

        **Output:**
        Return only the report content in Markdown format, starting with the main title.
        """

    def _format_community_details(self, communities: List[Dict]) -> str:
        """Formats the list of community details into a string for the prompt."""
        details = ""
        for i, community in enumerate(communities, 1):
            details += (
                f"- Community {i}:\n"
                f"  - Name: \"{community['community_name']}\"\n"
                f"  - Summary: \"{community['summary']}\"\n"
                f"  - Keywords: \"{community['keywords']}\"\n"
            )
        return details

if __name__ == '__main__':
    # Example of how to use the ReportService for a brief report
    try:
        print("Initializing Report Service...")
        service = ReportService()
        print("Service initialized.")
        
        print("\nGenerating brief report for 'demo' note...")
        gen_outline, gen_report = service.generate_brief_report(note_id="demo", note_title="Demo Brief Report")
        
        print("\n--- Generated Outline ---")
        print(json.dumps(gen_outline, indent=2, ensure_ascii=False))

        print("\n--- Generated Report (first 400 chars) ---")
        print(gen_report[:400] + "...")

    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Please ensure you have run the graph construction for the 'demo' dataset first.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
