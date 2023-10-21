from sklearn.metrics.cluster import normalized_mutual_info_score

def community_to_vector(communities,N):
    vector_of_communities = []
    for node in range(N):
        for el in range(len(communities)):
            if node in communities[el]:
                vector_of_communities.append(el)
    return vector_of_communities


