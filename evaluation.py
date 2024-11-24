

def generate_top_5_words(G, attribute_partition, top_n=5):
    """
    Generate the top n words for each partition in the attribute partition.

    Args:
    G: nx.Graph
    attribute_partition: name of attribute in graph, which is a dictionary of partition names and node ids
    top_n: int

    Returns:
    top_5_words: dict
    """
    top_5_words = {}
    for partition in attribute_partition:
        nodes = attribute_partition[partition]
        word_freq = {}
        for node in nodes:
            word = G.nodes[node]['word']
            if word in word_freq:
                word_freq[word] += 1
            else:
                word_freq[word] = 1
        top_5_words[partition] = sorted(word_freq, key=word_freq.get, reverse=True)[:top_n]
    return top_5_words