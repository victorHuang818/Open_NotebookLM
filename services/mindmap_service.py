import json
from typing import Dict, List

# Temporarily add youtu-graphrag to python path for imports
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from notebookLM.tools.retrieval_tools import RetrievalTools
from utils.call_llm_api import LLMCompletionCall
import json_repair

class MindMapService:
    """
    Service responsible for generating a mind map from a note's knowledge graph.
    """
    def __init__(self):
        self.llm_client = LLMCompletionCall()

    def generate_mindmap(self, note_id: str, note_title: str, max_layers: int = 4) -> Dict:
        """
        Generates a mind map for a given note.

        Args:
            note_id: The ID of the note.
            note_title: The title of the note, to be used as the root of the mind map.
            max_layers: The maximum desired depth of the mind map.

        Returns:
            A dictionary representing the mind map in a hierarchical JSON format.
        """
        try:
            # 1. Initialize tools to get knowledge overview
            retrieval_tools = RetrievalTools(note_id=note_id)
            communities = retrieval_tools.get_knowledge_overview()

            if not communities:
                return {"name": note_title, "children": [{"name": "No communities found to generate a mind map."}]}

            # 2. Build the prompt for the LLM
            prompt = self._build_mindmap_prompt(note_title, communities, max_layers)

            # 3. Call the LLM to get the structured mind map
            response_text = self.llm_client.call_api(prompt)

            # 4. Parse and return the JSON response
            mindmap_json = json_repair.loads(response_text)
            return mindmap_json

        except Exception as e:
            print(f"Error generating mind map for note '{note_id}': {e}")
            # Return a valid mind map structure with the error message
            return {"name": note_title, "children": [{"name": f"An error occurred: {e}"}]}

    def _build_mindmap_prompt(self, title: str, communities: List[Dict], max_layers: int) -> str:
        """
        Builds the prompt to instruct the LLM to generate a mind map.
        """
        community_details = ""
        for i, community in enumerate(communities, 1):
            community_details += (
                f"- Community {i}:\n"
                f"  - Name: \"{community['community_name']}\"\n"
                f"  - Summary: \"{community['summary']}\"\n"
                f"  - Keywords: \"{community['keywords']}\"\n"
            )

        prompt = f"""
        You are an expert at synthesizing information and creating structured, hierarchical mind maps.
        Based on the following knowledge communities extracted from a set of documents, please generate a mind map in JSON format.

        **Instructions:**
        1. The root of the mind map should be "{title}".
        2. Organize the communities into a logical hierarchy. Group similar or related communities under common parent nodes.
        3. The final mind map should NOT exceed {max_layers} layers in depth.
        4. The leaf nodes of the mind map should represent the individual communities. The name of the leaf node should be the community name. You can optionally include the community's keywords or summary in a 'value' field.
        5. Ensure the output is ONLY a valid JSON object that follows the specified format. Do not include any explanatory text or markdown formatting.

        **Knowledge Communities:**
        {community_details}

        **Required JSON Output Format:**
        {{
          "name": "root_node_name",
          "children": [
            {{
              "name": "layer_1_topic_A",
              "children": [
                {{ "name": "community_name_1", "value": "keywords/summary_1" }},
                {{ "name": "community_name_2", "value": "keywords/summary_2" }}
              ]
            }},
            {{
              "name": "layer_1_topic_B",
              "children": [
                {{ "name": "community_name_3", "value": "keywords/summary_3" }}
              ]
            }}
          ]
        }}

        Now, generate the mind map JSON for the title "{title}".
        """
        return prompt

if __name__ == '__main__':
    # Example of how to use the MindMapService
    try:
        print("Initializing Mind Map Service...")
        service = MindMapService()
        print("Service initialized.")
        
        print("\nGenerating mind map for 'demo' note...")
        # Assuming the note 'demo' has a title 'Demo Note'
        mindmap = service.generate_mindmap(note_id="demo", note_title="Demo Note")
        
        print("\n--- Generated Mind Map JSON ---")
        print(json.dumps(mindmap, indent=2, ensure_ascii=False))

    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Please ensure you have run the graph construction for the 'demo' dataset first.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
