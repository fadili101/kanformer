"""
KANFormer - Knowledge Graph Construction
========================================
This module handles the construction and embedding of educational knowledge graphs.
"""

import numpy as np
import pandas as pd
import networkx as nx
from sklearn.metrics import mutual_information_regression
import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict

class KnowledgeGraph:
    """Educational domain knowledge graph"""
    
    def __init__(self, feature_names, n_classes):
        """
        Initialize knowledge graph
        
        Parameters:
        -----------
        feature_names : list
            Names of features
        n_classes : int
            Number of target classes
        """
        self.feature_names = feature_names
        self.n_classes = n_classes
        self.graph = nx.DiGraph()
        self.entities = []
        self.relations = []
        self.triples = []
        
    def construct_from_data(self, X, y, correlation_threshold=0.3):
        """
        Construct knowledge graph from data
        
        Parameters:
        -----------
        X : numpy array
            Feature matrix
        y : numpy array
            Target labels
        correlation_threshold : float
            Threshold for creating edges
        """
        print("\nConstructing Knowledge Graph...")
        print("=" * 50)
        
        # 1. Add feature nodes
        print("1. Adding feature nodes...")
        for i, feature in enumerate(self.feature_names):
            self.graph.add_node(f"feature_{i}", 
                              name=feature, 
                              type="feature",
                              index=i)
            self.entities.append(f"feature_{i}")
        
        # 2. Add class nodes
        print("2. Adding class nodes...")
        for c in range(self.n_classes):
            self.graph.add_node(f"class_{c}", 
                              name=f"Class_{c}", 
                              type="class",
                              index=c)
            self.entities.append(f"class_{c}")
        
        # 3. Add correlation-based edges between features
        print("3. Computing feature correlations...")
        n_features = X.shape[1]
        for i in range(n_features):
            for j in range(i + 1, n_features):
                corr = np.corrcoef(X[:, i], X[:, j])[0, 1]
                if abs(corr) > correlation_threshold:
                    weight = abs(corr)
                    self.graph.add_edge(f"feature_{i}", 
                                      f"feature_{j}",
                                      relation="correlates_with",
                                      weight=weight)
                    self.triples.append((f"feature_{i}", 
                                       "correlates_with", 
                                       f"feature_{j}",
                                       weight))
                    self.relations.append("correlates_with")
        
        # 4. Add feature-to-class edges based on mutual information
        print("4. Computing feature-class relationships...")
        for i in range(n_features):
            for c in range(self.n_classes):
                # Mutual information between feature and class
                mask = (y == c)
                if mask.sum() > 10:  # Need sufficient samples
                    mi = self._compute_mi(X[:, i], mask.astype(int))
                    if mi > 0.05:  # Threshold for relevance
                        self.graph.add_edge(f"feature_{i}",
                                          f"class_{c}",
                                          relation="predicts",
                                          weight=mi)
                        self.triples.append((f"feature_{i}",
                                           "predicts",
                                           f"class_{c}",
                                           mi))
                        self.relations.append("predicts")
        
        # 5. Add domain knowledge edges
        print("5. Adding domain knowledge...")
        self._add_domain_knowledge()
        
        print(f"\nKnowledge Graph Statistics:")
        print(f"  Nodes: {self.graph.number_of_nodes()}")
        print(f"  Edges: {self.graph.number_of_edges()}")
        print(f"  Unique relations: {len(set(self.relations))}")
        print("=" * 50)
        
    def _compute_mi(self, x, y):
        """Compute mutual information"""
        from sklearn.metrics import mutual_info_score
        
        # Discretize continuous variable
        x_discrete = pd.qcut(x, q=5, labels=False, duplicates='drop')
        return mutual_info_score(x_discrete, y)
    
    def _add_domain_knowledge(self):
        """Add predefined domain knowledge relationships"""
        
        # Define common educational relationships
        domain_rules = [
            # Academic performance relationships
            ("previous_gpa", "prerequisite", "midterm_score"),
            ("attendance_rate", "influences", "previous_gpa"),
            ("study_time_weekly", "influences", "midterm_score"),
            ("assignment_submission_rate", "influences", "previous_gpa"),
            
            # Behavioral chains
            ("motivation_score", "influences", "study_time_weekly"),
            ("peer_interaction_index", "influences", "forum_participation"),
            
            # Socioeconomic factors
            ("parent_education", "influences", "previous_gpa"),
            ("internet_access", "enables", "forum_participation"),
        ]
        
        # Try to add domain rules if features exist
        feature_dict = {name: idx for idx, name in enumerate(self.feature_names)}
        
        for head, relation, tail in domain_rules:
            if head in feature_dict and tail in feature_dict:
                head_node = f"feature_{feature_dict[head]}"
                tail_node = f"feature_{feature_dict[tail]}"
                
                if not self.graph.has_edge(head_node, tail_node):
                    self.graph.add_edge(head_node, tail_node,
                                      relation=relation,
                                      weight=0.8)
                    self.triples.append((head_node, relation, tail_node, 0.8))
                    self.relations.append(relation)
    
    def get_adjacency_matrix(self):
        """Get weighted adjacency matrix"""
        n_nodes = self.graph.number_of_nodes()
        adj_matrix = np.zeros((n_nodes, n_nodes))
        
        node_to_idx = {node: idx for idx, node in enumerate(self.graph.nodes())}
        
        for edge in self.graph.edges(data=True):
            i = node_to_idx[edge[0]]
            j = node_to_idx[edge[1]]
            weight = edge[2].get('weight', 1.0)
            adj_matrix[i, j] = weight
            
        return adj_matrix
    
    def visualize(self, save_path='knowledge_graph.png'):
        """Visualize knowledge graph (requires matplotlib)"""
        try:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(15, 10))
            pos = nx.spring_layout(self.graph, k=2, iterations=50)
            
            # Draw nodes
            feature_nodes = [n for n, d in self.graph.nodes(data=True) 
                           if d.get('type') == 'feature']
            class_nodes = [n for n, d in self.graph.nodes(data=True) 
                         if d.get('type') == 'class']
            
            nx.draw_networkx_nodes(self.graph, pos, 
                                 nodelist=feature_nodes,
                                 node_color='lightblue',
                                 node_size=500,
                                 label='Features')
            nx.draw_networkx_nodes(self.graph, pos,
                                 nodelist=class_nodes,
                                 node_color='lightcoral',
                                 node_size=800,
                                 label='Classes')
            
            # Draw edges
            nx.draw_networkx_edges(self.graph, pos, alpha=0.3)
            
            plt.legend()
            plt.title("Educational Knowledge Graph")
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\nGraph visualization saved to {save_path}")
            plt.close()
        except ImportError:
            print("\nMatplotlib not available for visualization")


class TransE(nn.Module):
    """TransE model for knowledge graph embedding"""
    
    def __init__(self, n_entities, n_relations, embedding_dim=64, margin=1.0):
        """
        Initialize TransE model
        
        Parameters:
        -----------
        n_entities : int
            Number of entities in KG
        n_relations : int
            Number of relation types
        embedding_dim : int
            Dimension of embeddings
        margin : float
            Margin for ranking loss
        """
        super(TransE, self).__init__()
        
        self.n_entities = n_entities
        self.n_relations = n_relations
        self.embedding_dim = embedding_dim
        self.margin = margin
        
        # Entity and relation embeddings
        self.entity_embeddings = nn.Embedding(n_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(n_relations, embedding_dim)
        
        # Initialize embeddings
        nn.init.xavier_uniform_(self.entity_embeddings.weight)
        nn.init.xavier_uniform_(self.relation_embeddings.weight)
        
        # Normalize embeddings
        self.entity_embeddings.weight.data = torch.nn.functional.normalize(
            self.entity_embeddings.weight.data, p=2, dim=1
        )
        
    def forward(self, heads, relations, tails):
        """
        Forward pass
        
        Parameters:
        -----------
        heads : torch.Tensor
            Head entity indices
        relations : torch.Tensor
            Relation indices
        tails : torch.Tensor
            Tail entity indices
        
        Returns:
        --------
        score : torch.Tensor
            Translation scores
        """
        head_embeds = self.entity_embeddings(heads)
        relation_embeds = self.relation_embeddings(relations)
        tail_embeds = self.entity_embeddings(tails)
        
        # TransE: h + r ≈ t
        # Score is negative distance (lower distance = higher score)
        score = head_embeds + relation_embeds - tail_embeds
        score = torch.norm(score, p=2, dim=1)
        
        return score
    
    def get_embeddings(self):
        """Get learned entity embeddings"""
        return self.entity_embeddings.weight.data.cpu().numpy()


class KGEmbedder:
    """Knowledge graph embedding trainer"""
    
    def __init__(self, knowledge_graph, embedding_dim=64):
        """
        Initialize KG embedder
        
        Parameters:
        -----------
        knowledge_graph : KnowledgeGraph
            Knowledge graph object
        embedding_dim : int
            Dimension of embeddings
        """
        self.kg = knowledge_graph
        self.embedding_dim = embedding_dim
        self.entity_to_id = {e: i for i, e in enumerate(self.kg.entities)}
        self.relation_to_id = {r: i for i, r in enumerate(set(self.kg.relations))}
        self.model = None
        
    def prepare_training_data(self):
        """Prepare training triples"""
        print("\nPreparing training data for TransE...")
        
        positive_triples = []
        for triple in self.kg.triples:
            head, rel, tail, weight = triple
            if head in self.entity_to_id and tail in self.entity_to_id:
                h_id = self.entity_to_id[head]
                r_id = self.relation_to_id.get(rel, 0)
                t_id = self.entity_to_id[tail]
                positive_triples.append([h_id, r_id, t_id, weight])
        
        self.positive_triples = np.array(positive_triples)
        print(f"  Prepared {len(self.positive_triples)} positive triples")
        
        return self.positive_triples
    
    def corrupt_triple(self, triple):
        """Generate negative sample by corrupting head or tail"""
        h, r, t, w = triple
        
        # Randomly corrupt head or tail
        if np.random.random() < 0.5:
            # Corrupt head
            h_corrupted = np.random.randint(0, len(self.entity_to_id))
            return [h_corrupted, r, t, w]
        else:
            # Corrupt tail
            t_corrupted = np.random.randint(0, len(self.entity_to_id))
            return [h, r, t_corrupted, w]
    
    def train(self, n_epochs=100, batch_size=128, learning_rate=0.01):
        """
        Train TransE model
        
        Parameters:
        -----------
        n_epochs : int
            Number of training epochs
        batch_size : int
            Batch size
        learning_rate : float
            Learning rate
        
        Returns:
        --------
        embeddings : numpy array
            Learned entity embeddings
        """
        print(f"\nTraining TransE model...")
        print(f"  Epochs: {n_epochs}")
        print(f"  Batch size: {batch_size}")
        print(f"  Learning rate: {learning_rate}")
        print("=" * 50)
        
        # Initialize model
        self.model = TransE(
            n_entities=len(self.entity_to_id),
            n_relations=len(self.relation_to_id),
            embedding_dim=self.embedding_dim,
            margin=1.0
        )
        
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Prepare data
        triples = self.prepare_training_data()
        n_triples = len(triples)
        
        # Training loop
        for epoch in range(n_epochs):
            total_loss = 0
            n_batches = 0
            
            # Shuffle triples
            np.random.shuffle(triples)
            
            for i in range(0, n_triples, batch_size):
                batch = triples[i:i + batch_size]
                
                # Generate corrupted triples
                corrupted_batch = np.array([self.corrupt_triple(t) for t in batch])
                
                # Convert to tensors
                pos_heads = torch.LongTensor(batch[:, 0])
                pos_rels = torch.LongTensor(batch[:, 1])
                pos_tails = torch.LongTensor(batch[:, 2])
                
                neg_heads = torch.LongTensor(corrupted_batch[:, 0])
                neg_rels = torch.LongTensor(corrupted_batch[:, 1])
                neg_tails = torch.LongTensor(corrupted_batch[:, 2])
                
                # Forward pass
                pos_scores = self.model(pos_heads, pos_rels, pos_tails)
                neg_scores = self.model(neg_heads, neg_rels, neg_tails)
                
                # Margin ranking loss
                loss = torch.mean(torch.relu(self.model.margin + pos_scores - neg_scores))
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Normalize embeddings
                self.model.entity_embeddings.weight.data = torch.nn.functional.normalize(
                    self.model.entity_embeddings.weight.data, p=2, dim=1
                )
                
                total_loss += loss.item()
                n_batches += 1
            
            avg_loss = total_loss / n_batches
            
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch + 1}/{n_epochs}, Loss: {avg_loss:.4f}")
        
        print("=" * 50)
        print("Training complete!")
        
        # Get embeddings
        embeddings = self.model.get_embeddings()
        
        return embeddings
    
    def get_feature_embeddings(self):
        """Get embeddings for features only"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        embeddings = self.model.get_embeddings()
        
        # Extract feature embeddings
        feature_embeddings = []
        for i, feature_name in enumerate(self.kg.feature_names):
            entity_name = f"feature_{i}"
            if entity_name in self.entity_to_id:
                idx = self.entity_to_id[entity_name]
                feature_embeddings.append(embeddings[idx])
        
        return np.array(feature_embeddings)


# Usage example
if __name__ == "__main__":
    # Create dummy data for demonstration
    np.random.seed(42)
    n_samples = 1000
    n_features = 20
    n_classes = 4
    
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, n_classes, n_samples)
    
    feature_names = [f"feature_{i}" for i in range(n_features)]
    
    # Build knowledge graph
    print("\n" + "="*70)
    print("KNOWLEDGE GRAPH CONSTRUCTION AND EMBEDDING")
    print("="*70)
    
    kg = KnowledgeGraph(feature_names, n_classes)
    kg.construct_from_data(X, y, correlation_threshold=0.3)
    
    # Train embeddings
    embedder = KGEmbedder(kg, embedding_dim=64)
    embeddings = embedder.train(n_epochs=50, batch_size=64, learning_rate=0.01)
    
    # Get feature embeddings
    feature_embeddings = embedder.get_feature_embeddings()
    print(f"\nFeature embeddings shape: {feature_embeddings.shape}")
    
    # Optional: Visualize graph
    # kg.visualize('kg_example.png')
    
    print("\n" + "="*70)
    print("Knowledge graph embeddings ready for KANFormer!")
    print("="*70)
