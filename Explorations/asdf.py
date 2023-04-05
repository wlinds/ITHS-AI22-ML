import torch.nn.functional as F
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel

# mean pooling from sbert
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    # First element of model_output contains all token embeddings

    input_mask_expanded = (attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float())
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

def load_model():
    tokenizer = AutoTokenizer.from_pretrained("intfloat/e5-large")
    model = AutoModel.from_pretrained("intfloat/e5-large").eval()
    return tokenizer, model

def generate_embeddings(tokenizer, model, documents):
    batch_dict = tokenizer(
        documents, 
        max_length=512,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )

    with torch.inference_mode():
        outputs = model(**batch_dict)
    return mean_pooling(outputs, batch_dict["attention_mask"]).cpu().numpy()

if __name__ == "__main__":
    tokenizer, model = load_model()
    sample_documents = [
        "The duck went to the restaurant."
        "The cat stole a telephoe."
        "Police found that a phone was missing, it was pretty suspecious."
        "Police brought dogs. The dogs were trained to find corpses."
    ]

    document_embeddings = generate_embeddings(tokenizer, model, [f'passage: {t}' for t in sample_documents])

    query = "query: Who stole the phone?"
    query_embedding = generate_embeddings(tokenizer, model, query)

    similarities = cosine_similarity(query_embedding, document_embeddings)
    best_match_index = np.argmax(similarities)
    best_match_document = sample_documents[best_match_index]

    print(similarities[0])
    print(best_match_document)
    print('Complete')