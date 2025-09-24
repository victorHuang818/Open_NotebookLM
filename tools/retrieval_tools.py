from typing import Dict, List
import networkx as nx

# Temporarily add youtu-graphrag to python path for imports
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from utils import graph_processor
from models.retriever.enhanced_kt_retriever import KTRetriever
from config import get_config

class RetrievalTools:
    """
    A collection of tools for the report generation agent that interface with the 
    youtu-graphrag knowledge engine (KTRetriever).
    """

    def __init__(self, note_id: str):
        """
        Initializes the retrieval tools for a specific note.

        Args:
            note_id: The identifier for the note, used to load the correct
                     knowledge graph and associated caches.
        """
        self.note_id = note_id
        self.config = get_config()
        
        # This assumes a naming convention where graph outputs are stored
        # with the note_id as the dataset name.
        graph_path = self.config.get_dataset_config(note_id).graph_output
        schema_path = self.config.get_dataset_config(note_id).schema_path

        if not os.path.exists(graph_path):
            raise FileNotFoundError(f"Knowledge graph for note '{note_id}' not found at {graph_path}")

        self.kt_retriever = KTRetriever(
            dataset=note_id,
            json_path=graph_path,
            schema_path=schema_path,
            config=self.config
        )
        # It's crucial to build indices if they are not already cached
        self.kt_retriever.build_indices()

    def get_knowledge_overview(self) -> List[Dict[str, str]]:
        """
        Provides a high-level overview of the knowledge base by returning 
        the name, summary, and keywords of each identified community.

        Returns:
            A list of dictionaries, where each dictionary represents a community.
        """
        overview_data = []
        graph = self.kt_retriever.graph
        
        community_nodes = [
            (node_id, node_data)
            for node_id, node_data in graph.nodes(data=True)
            if node_data.get("label") == "community"
        ]

        for comm_id, comm_data in community_nodes:
            properties = comm_data.get("properties", {})
            
            # Extract keywords associated with this community
            keywords = []
            for neighbor in graph.predecessors(comm_id):
                neighbor_data = graph.nodes[neighbor]
                if neighbor_data.get("label") == "keyword":
                    keywords.append(neighbor_data.get("properties", {}).get("name", ""))

            overview_data.append({
                "community_name": properties.get("name", "Unnamed Community"),
                "summary": properties.get("description", "No summary available."),
                "keywords": ", ".join(keywords) if keywords else "N/A"
            })
            
        return overview_data

    def search_topic_details(self, topic_query: str, search_depth: str = "deep") -> str:
        """
        Searches for detailed information on a specific topic within the knowledge base.

        Args:
            topic_query: The specific topic or question to search for.
            search_depth: 'deep' returns raw triples and chunks, 'summary' returns an LLM-generated summary.

        Returns:
            A string containing either the detailed search results or a concise summary.
        """
        if not topic_query:
            return "Error: Please provide a topic to search for."

        try:
            # The core retrieval is done here
            results, _ = self.kt_retriever.process_retrieval_results(question=topic_query)
            
            triples = results.get('triples', [])
            chunks = results.get('chunk_contents', [])

            if not triples and not chunks:
                return f"No detailed information found for the topic: '{topic_query}'"

            context = "=== Retrieved Triples ===\n" + "\n".join(triples)
            context += "\n\n=== Retrieved Text Chunks ===\n" + "\n\n".join(chunks)

            if search_depth == "summary":
                # Use LLM to summarize the deep search results
                prompt = f"""
                Based on the following retrieved information, please provide a concise summary that answers the query: "{topic_query}".

                Retrieved Context:
                {context}

                Concise Summary:
                """
                summary = self.kt_retriever.llm_client.call_api(prompt)
                return summary
            else:
                # Return the raw, detailed context
                return context

        except Exception as e:
            print(f"Error during topic search for '{topic_query}': {e}")
            return f"An error occurred while searching for topic '{topic_query}': {e}"

    def find_connections_between_topics(self, topic_a: str, topic_b: str) -> str:
        """
        Finds and describes the shortest path between the core entities of two topics in the knowledge graph.

        Args:
            topic_a: The first topic.
            topic_b: The second topic.

        Returns:
            A string describing the connection, or a message indicating no direct connection was found.
        """
        try:
            # 1. Use retrieval to find the most relevant nodes for each topic
            results_a, _ = self.kt_retriever.process_retrieval_results(question=topic_a, top_k=1)
            results_b, _ = self.kt_retriever.process_retrieval_results(question=topic_b, top_k=1)

            if not results_a.get('triples') or not results_b.get('triples'):
                return f"Could not establish a clear entity for one of the topics: '{topic_a}' or '{topic_b}'."

            # A simple way to get the primary entity is to look at the first node in the top triple
            # This is a heuristic and could be improved
            source_node_name = results_a['triples'][0].split(',')[0].strip('( ')
            target_node_name = results_b['triples'][0].split(',')[0].strip('( ')

            # Find the node IDs from their names
            name_to_id_map = {data['properties'].get('name'): node_id 
                              for node_id, data in self.kt_retriever.graph.nodes(data=True) 
                              if data.get('properties')}
            
            source_node_id = name_to_id_map.get(source_node_name)
            target_node_id = name_to_id_map.get(target_node_name)

            if not source_node_id or not target_node_id:
                return f"Could not find one of the core entities in the graph: '{source_node_name}' or '{target_node_name}'."
            
            if source_node_id == target_node_id:
                return f"The topics '{topic_a}' and '{topic_b}' appear to refer to the same core entity: '{source_node_name}'."

            # 2. Use networkx to find the shortest path
            try:
                path_nodes = nx.shortest_path(self.kt_retriever.graph.to_undirected(), source=source_node_id, target=target_node_id)
                
                # 3. Describe the path
                path_description = f"Connection found between '{source_node_name}' and '{target_node_name}':\n"
                path_description += f"  - {source_node_name}\n"
                
                for i in range(len(path_nodes) - 1):
                    u, v = path_nodes[i], path_nodes[i+1]
                    # In a MultiDiGraph, there can be multiple edges. We'll take the first one.
                    edge_data = self.kt_retriever.graph.get_edge_data(u, v)
                    relation = "is connected to" # Default
                    if edge_data:
                        relation = list(edge_data.values())[0].get('relation', relation)

                    v_name = self.kt_retriever.graph.nodes[v]['properties'].get('name', v)
                    path_description += f"    --[{relation}]--> {v_name}\n"
                
                return path_description

            except nx.NetworkXNoPath:
                return f"No direct path found in the knowledge graph between the core entities of '{topic_a}' and '{topic_b}'."

        except Exception as e:
            print(f"Error finding connections: {e}")
            return f"An error occurred while finding connections: {e}"


if __name__ == '__main__':
    # This is an example of how to use the tool.
    # Note: You need to have a graph pre-built for the 'demo' dataset.
    try:
        print("Initializing retrieval tools for 'demo' note...")
        demo_tools = RetrievalTools(note_id="demo")
        print("Tools initialized.")
        
        print("\nFetching knowledge overview...")
        overview = demo_tools.get_knowledge_overview()
        
        if overview:
            print(f"Successfully retrieved {len(overview)} communities.")
            for i, community in enumerate(overview, 1):
                print(f"\n--- Community {i} ---")
                print(f"  Name: {community['community_name']}")
                print(f"  Summary: {community['summary']}")
                print(f"  Keywords: {community['keywords']}")
        else:
            print("No communities found in the knowledge graph.")

        print("\n" + "="*50)
        print("Testing Topic Search...")
        topic_query = "community detection" # A query likely to have results in the demo graph
        print(f"Searching for topic: '{topic_query}' (deep dive)")
        details_deep = demo_tools.search_topic_details(topic_query, search_depth="deep")
        print("\n--- Deep Dive Results ---")
        print(details_deep[:1000] + "..." if len(details_deep) > 1000 else details_deep)
        
        print(f"\nSearching for topic: '{topic_query}' (summary)")
        details_summary = demo_tools.search_topic_details(topic_query, search_depth="summary")
        print("\n--- Summary Results ---")
        print(details_summary)

        print("\n" + "="*50)
        print("Testing Connection Finder...")
        # These topics are examples; their effectiveness depends on the content of demo_corpus.json
        topic_1 = "community"
        topic_2 = "knowledge graph"
        print(f"Finding connection between '{topic_1}' and '{topic_2}'...")
        connection = demo_tools.find_connections_between_topics(topic_1, topic_2)
        print("\n--- Connection Results ---")
        print(connection)


    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Please ensure you have run the graph construction for the 'demo' dataset first.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
