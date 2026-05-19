# node_management.py

class Node:
    def __init__(self, node_id: str):
        """
        Represents a node in the network.

        Args:
            node_id (str): The unique identifier for the node.
        """
        self.id = node_id
        self.active = True  # Node is active by default.
        self.interaction_history = []  # List to store dictionaries of interaction details.
        self.endorsement_history = []  # List to store dictionaries of endorsement events.
        self.reputation_history = []   # List to store reputation snapshots over time.

    def mark_inactive(self):
        """Marks this node as inactive."""
        self.active = False

    def mark_active(self):
        """Marks this node as active."""
        self.active = True

    def add_interaction(self, interaction: dict):
        """
        Adds an interaction event to the node's history.

        Args:
            interaction (dict): A dictionary containing interaction details,
                                e.g., {"src": ..., "dst": ..., "rating": ..., "timestamp": ..., "month": ..., "reputation_change": ...}
        """
        self.interaction_history.append(interaction)

    def add_endorsement_change(self, endorsement_change: dict):
        """
        Adds an endorsement event/change to the node's history.

        Args:
            endorsement_change (dict): A dictionary containing endorsement change details,
                                       e.g., {"action": "added" or "removed",
                                              "counterpart": <other node id>,
                                              "old_weight": <old weight>,
                                              "new_weight": <new weight>,
                                              "trigger": <interaction details or reason>}
        """
        self.endorsement_history.append(endorsement_change)

    def add_reputation_snapshot(self, reputation_value: float):
        """
        Adds a reputation snapshot to the node's reputation history.
        
        Args:
            reputation_value (float): The reputation score at a given time.
        """
        self.reputation_history.append(reputation_value)

    def __repr__(self):
        return (f"Node(id={self.id}, active={self.active}, "
                f"interactions={len(self.interaction_history)}, "
                f"endorsements={len(self.endorsement_history)}, "
                f"reputation_history={self.reputation_history})")


class NodeManager:
    def __init__(self):
        """
        Manages network nodes.
        
        Nodes are stored in a dictionary mapping node_id to a Node instance.
        """
        self.nodes = {}  # Dict[str, Node]

    def load_nodes(self, node_list):
        """
        Loads an initial list of nodes into the manager.
        Each node is created as active.

        Args:
            node_list (list): A list of node IDs.
        """
        for node_id in node_list:
            if node_id not in self.nodes:
                self.nodes[node_id] = Node(node_id)
            else:
                # If the node already exists (perhaps reactivation), mark as active.
                self.nodes[node_id].mark_active()

    def add_node(self, node_id: str):
        """
        Adds a new node to the network. If the node already exists and is inactive,
        it will be reactivated.
        
        Args:
            node_id (str): The node identifier to add.
        """
        if node_id in self.nodes:
            node = self.nodes[node_id]
            if not node.active:
                node.mark_active()
                print(f"Reactivated node {node_id}.")
            else:
                print(f"Node {node_id} already active.")
        else:
            self.nodes[node_id] = Node(node_id)
            print(f"Added new node {node_id}.")

    def remove_node(self, node_id: str):
        """
        Marks a node as inactive (rather than deleting it) to preserve historical data.
        
        Args:
            node_id (str): The node identifier to mark as inactive.
        """
        if node_id in self.nodes:
            self.nodes[node_id].mark_inactive()
            print(f"Node {node_id} marked as inactive.")
        else:
            print(f"Node {node_id} not found.")

    def display_active_nodes(self):
        """
        Displays and returns a list of IDs for active nodes.
        """
        active_nodes = [node_id for node_id, node in self.nodes.items() if node.active]
        print("Active nodes:", active_nodes)
        return active_nodes

    def display_inactive_nodes(self):
        """
        Displays and returns a list of IDs for inactive nodes.
        """
        inactive_nodes = [node_id for node_id, node in self.nodes.items() if not node.active]
        print("Inactive nodes:", inactive_nodes)
        return inactive_nodes

    def display_all_nodes(self):
        """
        Displays and returns detailed information for all nodes,
        including interaction and endorsement histories.
        """
        for node_id, node in self.nodes.items():
            print(node)
        return list(self.nodes.items())

    def count_nodes(self):
        """Returns the total number of nodes managed."""
        return len(self.nodes)

    def count_active_nodes(self):
        """Returns the number of active nodes."""
        return len([node for node in self.nodes.values() if node.active])

    def count_inactive_nodes(self):
        """Returns the number of inactive nodes."""
        return len([node for node in self.nodes.values() if not node.active])


# Example usage:
if __name__ == "__main__":
    # Create a NodeManager instance.
    nm = NodeManager()
    
    # Load an initial list of nodes.
    node_list = ['A', 'B', 'C', 'D']
    nm.load_nodes(node_list)
    
    print("All nodes loaded:")
    nm.display_all_nodes()
    
    # Add a new node 'E'
    nm.add_node('E')
    
    # Record an interaction event in node 'A'
    interaction_event = {
        "src": "A",
        "dst": "B",
        "rating": 1,
        "timestamp": 1609459200,
        "month": 1,
        "reputation_change": 0.05
    }
    nm.nodes["A"].add_interaction(interaction_event)
    
    # Record an endorsement update event in node 'A'
    endorsement_event = {
        "action": "added",
        "counterpart": "C",
        "old_weight": 0.0,
        "new_weight": 0.5,
        "trigger": "Interaction with B increased trust."
    }
    nm.nodes["A"].add_endorsement_change(endorsement_event)
    
    # Record a reputation snapshot for node 'A'
    nm.nodes["A"].add_reputation_snapshot(0.52)
    
    # Display active nodes.
    nm.display_active_nodes()
    
    # Remove node 'C' (mark inactive)
    nm.remove_node("C")
    
    print("\nFinal node statuses:")
    nm.display_all_nodes()
