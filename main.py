import random
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import argparse

amino_acid_to_id = {
    "A": 0, "C": 1, "D": 2, "E": 3, "F": 4, "G": 5, "H": 6, "I": 7, "K": 8, "L": 9,
    "M": 10, "N": 11, "P": 12, "Q": 13, "R": 14, "S": 15, "T": 16, "V": 17, "W": 18, "Y": 19,
    "-": 20
}

amino_acids = "ACDEFGHIKLMNPQRSTVWY"

def introduce_mutations(sequence, mutation_rate=0.1, insertion_rate=0.05, deletion_rate=0.05):
    modified_sequence = []
    for amino_acid in sequence:
        if random.random() < deletion_rate:
            continue
        if random.random() < mutation_rate:
            amino_acid = random.choice(amino_acids)
        modified_sequence.append(amino_acid)
        if random.random() < insertion_rate:
            modified_sequence.append(random.choice(amino_acids))
    return ''.join(modified_sequence)

def simulate_msa(input_sequence, num_sequences=10, mutation_rate=0.1, insertion_rate=0.05, deletion_rate=0.05):
    msa = [input_sequence]
    for _ in range(num_sequences - 1):
        mutated_sequence = introduce_mutations(input_sequence, mutation_rate, insertion_rate, deletion_rate)
        msa.append(mutated_sequence)
    return msa

def convert_to_tensor(msa, amino_acid_to_id):
    num_sequences = len(msa)
    sequence_length = max(len(seq) for seq in msa)
    msa_tensor = torch.full((num_sequences, sequence_length), amino_acid_to_id["-"], dtype=torch.long)
    for i, sequence in enumerate(msa):
        for j, amino_acid in enumerate(sequence):
            msa_tensor[i, j] = amino_acid_to_id.get(amino_acid, amino_acid_to_id["-"])
    return msa_tensor

def prepare_msa_tensor(input_sequence):
    msa = simulate_msa(input_sequence)
    msa_tensor = convert_to_tensor(msa, amino_acid_to_id)
    return msa_tensor

class MSAEmbeddingModel(nn.Module):
    def __init__(self, num_amino_acids, embedding_dim, nhead, num_layers):
        super(MSAEmbeddingModel, self).__init__()
        self.embedding = nn.Embedding(num_amino_acids, embedding_dim)
        encoder_layers = TransformerEncoderLayer(embedding_dim, nhead)
        self.transformer = TransformerEncoder(encoder_layers, num_layers)
        
    def forward(self, msa_tensor):
        embedded = self.embedding(msa_tensor)
        embedded = embedded.permute(1, 0, 2)
        msa_embeddings = self.transformer(embedded)
        msa_embeddings = msa_embeddings.permute(1, 0, 2)
        return msa_embeddings

def main():
    parser = argparse.ArgumentParser(description="Generate MSA embeddings from a protein sequence.")
    parser.add_argument("sequence", type=str, help="Input protein sequence (e.g., 'ACDEFGHIKLMNPQRSTVWY')")
    args = parser.parse_args()

    num_amino_acids = 21
    embedding_dim = 64
    nhead = 8
    num_layers = 4

    model = MSAEmbeddingModel(num_amino_acids, embedding_dim, nhead, num_layers)

    msa_tensor = prepare_msa_tensor(args.sequence)
    print("MSA Tensor:\n", msa_tensor)

    msa_embeddings = model(msa_tensor)
    print("MSA Embeddings:\n", msa_embeddings)

if __name__ == "__main__":
    main()
