import re
from collections import defaultdict
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
import networkx as nx

# Clean text
def clean_text(text, stop_words):
    """
    Cleans the input text by removing non-alphabet characters and stopwords.
    """
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Keep only letters and spaces
    words = text.lower().split()
    cleaned_words = [word for word in words if word not in stop_words]
    return cleaned_words


def generate_wordcloud(G, communities, stop_words=set()):
    """
    Generate and display word clouds for each community in the graph.

    Parameters:
        G (nx.Graph): The graph containing nodes with text attributes.
        communities (dict): A dictionary with partition IDs as keys and lists of nodes as values.
        stop_words (set): A set of stopwords to exclude from the word clouds.
    """
    
    # Dictionary to store the combined text for each community
    community_texts = defaultdict(list)
    
    # Aggregate text data for each community
    for partition_id, nodes in communities.items():
        for node in nodes:
            text = G.nodes[node].get("course_text", "")
            if text:
                cleaned_words = clean_text(text, stop_words)
                community_texts[partition_id].append(" ".join(cleaned_words))
    
    # Generate and display word clouds for each community
    for partition_id, documents in community_texts.items():
        # Combine all text documents into a single string
        combined_text = " ".join(documents)
        
        # Calculate TF-IDF for the community's text
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([combined_text])
        feature_names = vectorizer.get_feature_names_out()
        tfidf_scores = tfidf_matrix.sum(axis=0).A1
        
        # Create a dictionary of terms and their aggregated TF-IDF scores
        aggregated_scores = dict(zip(feature_names, tfidf_scores))
        
        # Generate the word cloud
        wordcloud = WordCloud(width=800, height=400, background_color="white").generate_from_frequencies(aggregated_scores)
        
        # Get the top 3 most connected courses in the partition
        node_degrees = [(node, G.degree(node)) for node in communities[partition_id]]
        top_3_nodes = sorted(node_degrees, key=lambda x: x[1], reverse=True)[:3]
        top_3_course_names = [G.nodes[node]["course_name"] for node, _ in top_3_nodes]
        title = "Top 3 Courses: " + ", ".join(top_3_course_names)
        
        # Display the word cloud
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.title(f"Word cloud for Community {partition_id} - {title}")
        plt.show()
