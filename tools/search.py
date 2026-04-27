def search(query: str) -> str:
    """A search tool """

    fact_dict  = {
        "capital of india": "New Delhi",
        "president of the united states": "Donald Trump",
        "age of the universe according to indian mythology": "1.97 billion years",
        "tallest mountain of india": "Kanchenjunga",
        "8 world wonders": "Great Wall of China, Petra, Colosseum, Chichen Itza, Machu Picchu, Taj Mahal, Christ the Redeemer, Pyramids of Giza",
        "emerging AI models of 2026": "GPT-5, Gemini 3, Claude 4, Llama 4, Mistral 4"
    }   
    query = query.lower()
    if query in fact_dict:
        return fact_dict[query]

    words_in_query = set(query.split())
    max_overlap_count = 0
    matched_best_key = None

    for key in fact_dict:
        words_in_key = set(key.split())
        overlap = len(words_in_query.intersection(words_in_key))

        if overlap > max_overlap_count:
            max_overlap_count = overlap
            matched_best_key = key

    if max_overlap_count >= 2:
        return fact_dict[matched_best_key]
            
    return "Could not find answer. Try rephrasing."