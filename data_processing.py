import numpy as np
import pandas as pd
import scipy.io as sio
import ast
from collections import Counter

def extract_tokens_from_string(token_string):
    """Extract tokens exactly as they appear in a formatted string like ['TEMPE', ',', 'Ariz', '.']."""
    # Use literal_eval to parse the list directly from the string representation
    tokens = ast.literal_eval(token_string)
    return tokens

def clean_token(token):
    """Keep tokens as they are for exact matching, only convert to lowercase for case-insensitivity."""
    return str(token).lower()  # Convert only to lowercase to maintain exact structure

def load_embeddings(embeddings_file):
    """Load word embeddings from CSV file with space-separated values."""
    df = pd.read_csv(embeddings_file)
    embeddings_dict = {}
    
    for _, row in df.iterrows():
        try:
            word = str(row['word']).strip().lower()
            if pd.isna(word) or word == '':  # Skip empty or NaN values
                continue
                
            # Convert space-separated string to numpy array
            embedding_str = row['embedding'].strip('[]" \n')
            embedding = np.fromstring(embedding_str, sep=' ')
            
            # Store embedding with lowercase key for consistency
            embeddings_dict[word] = embedding
                
        except Exception as e:
            print(f"Error processing embedding row: {e}")
            continue
    
    # Define <start> based on 'start' embedding if available; otherwise, use zeros
    #embeddings_dict['<start>'] = embeddings_dict.get('start', np.zeros(64))
    embeddings_dict['<start>'] = np.zeros(64)
    
    # Define <unknown> based on 'unknown' embedding if available; otherwise, use zeros
    #embeddings_dict['<unknown>'] = embeddings_dict.get('unknown', np.zeros(64))
    embeddings_dict['<unknown>'] = np.zeros(64)
    
    return embeddings_dict

def check_embedding_coverage(tokens, embeddings_dict):
    """Check which tokens are missing from embeddings."""
    missing_tokens = set()
    token_counts = Counter()
    original_to_cleaned = {}
    
    for token in tokens:
        clean_tok = clean_token(token)  # Clean and lowercase token for case-insensitivity
        token_counts[clean_tok] += 1
        original_to_cleaned[token] = clean_tok
        if clean_tok not in embeddings_dict:
            missing_tokens.add(token)
    
    return missing_tokens, token_counts, original_to_cleaned

def parse_tokens_and_tags(row):
    """Parse tokens and POS tags from string representations."""
    tokens = extract_tokens_from_string(row['tokens'])
    pos_tags = ast.literal_eval(row['pos_tags'])
    return tokens, pos_tags

def convert_pos_tags(tag):
    """Convert POS tags to 4 classes."""
    # Noun tags
    if tag in [22, 23, 24, 25, 26, 27, 28]:
        return 1
    # Verb tags
    elif tag in [37, 38, 39, 40, 41, 42]:
        return 2
    # Adjective/Adverb tags
    elif tag in [16, 17, 18, 32, 33, 34]:
        return 3
    # Others
    else:
        return 4

def process_sequence(tokens, tags, embeddings_dict, sequence_length=4):
    """Process a sequence of tokens into overlapping windows."""
    processed_sequences = []
    processed_tags = []
    
    # Retrieve the <start> and <unknown> embeddings
    padding_start_embedding = embeddings_dict['<start>']
    unknown_embedding = embeddings_dict['<unknown>']
    
    # Generate padded sequences
    for i in range(len(tokens)):
        # Generate the current window with <start> as needed
        sequence = ['<start>'] * max(0, sequence_length - (i + 1)) + tokens[max(0, i - sequence_length + 1): i + 1]
        sequence_tags = [4] * max(0, sequence_length - (i + 1)) + [convert_pos_tags(tags[j]) for j in range(max(0, i - sequence_length + 1), i + 1)]
        
        # Ensure the sequence length is exactly 4
        sequence_embeddings = np.array([
            embeddings_dict.get(clean_token(tok), unknown_embedding)  # Use unknown embedding if token is missing
            for tok in sequence
        ])
        
        # Append the embedding sequence and tag
        processed_sequences.append(sequence_embeddings)
        processed_tags.append(sequence_tags[-1])  # Use the tag of the last token in the window
        
    return processed_sequences, processed_tags

def process_dataset(data_file, embeddings_dict):
    """Process entire dataset and check embedding coverage."""
    df = pd.read_csv(data_file)
    all_sequences = []
    all_tags = []
    all_tokens = []
    
    for _, row in df.iterrows():
        tokens, pos_tags = parse_tokens_and_tags(row)
        
        # Verify token and tag length consistency
        if len(tokens) != len(pos_tags):
            print(f"Warning: Mismatch in row with tokens: {tokens} and pos_tags: {pos_tags}")
            continue  # Skip this row or handle it as needed
        
        all_tokens.extend(tokens)
        sequences, tags = process_sequence(tokens, pos_tags, embeddings_dict)
        all_sequences.extend(sequences)
        all_tags.extend(tags)
    
    missing_tokens, token_counts, original_to_cleaned = check_embedding_coverage(all_tokens, embeddings_dict)
    
    return np.array(all_sequences), np.array(all_tags), missing_tokens, token_counts, original_to_cleaned

# Main processing
def main():
    try:
        # Load embeddings
        print("Loading embeddings...")
        embeddings_dict = load_embeddings('embeddings.csv')
        print(f"Loaded {len(embeddings_dict)} embeddings")
        
        # Process each dataset
        print("\nProcessing training data...")
        train_tokens, train_tags, train_missing, train_counts, train_cleaned = process_dataset('train_data.csv', embeddings_dict)
        print(f"Training set unique tokens: {len(set(train_counts.keys()))}")
        print(f"Training set missing embeddings: {len(train_missing)}")
        if train_missing:
            print("Sample of missing tokens in training set (first 5, with transformations):")
            for token in list(train_missing)[:5]:
                print(f"Original: {token} -> Cleaned: {train_cleaned[token]}")
        
        print("\nProcessing validation data...")
        valid_tokens, valid_tags, valid_missing, valid_counts, valid_cleaned = process_dataset('valid_data.csv', embeddings_dict)
        print(f"Validation set unique tokens: {len(set(valid_counts.keys()))}")
        print(f"Validation set missing embeddings: {len(valid_missing)}")
        if valid_missing:
            print("Sample of missing tokens in validation set (first 5, with transformations):")
            for token in list(valid_missing)[:5]:
                print(f"Original: {token} -> Cleaned: {valid_cleaned[token]}")
        
        print("\nProcessing test data...")
        test_tokens, test_tags, test_missing, test_counts, test_cleaned = process_dataset('test_data.csv', embeddings_dict)
        print(f"Test set unique tokens: {len(set(test_counts.keys()))}")
        print(f"Test set missing embeddings: {len(test_missing)}")
        if test_missing:
            print("Sample of missing tokens in test set (first 5, with transformations):")
            for token in list(test_missing)[:5]:
                print(f"Original: {token} -> Cleaned: {test_cleaned[token]}")
        
        # Calculate overall statistics
        all_missing = train_missing | valid_missing | test_missing
        all_tokens = set(train_counts.keys()) | set(valid_counts.keys()) | set(test_counts.keys())
        
        print("\nOverall Statistics:")
        print(f"Total unique tokens across all sets: {len(all_tokens)}")
        print(f"Total tokens missing embeddings: {len(all_missing)}")
        print(f"Embedding coverage: {((len(all_tokens) - len(all_missing)) / len(all_tokens)) * 100:.2f}%")
        
        # Save to MATLAB format
        print("\nSaving to MATLAB format...")
        sio.savemat('processed_data.mat', {
            'train_tokens': train_tokens,
            'train_tags': train_tags,
            'valid_tokens': valid_tokens,
            'valid_tags': valid_tags,
            'test_tokens': test_tokens,
            'test_tags': test_tags
        })
        
        print("\nData shapes:")
        print(f"Train tokens: {train_tokens.shape}, Train tags: {train_tags.shape}")
        print(f"Valid tokens: {valid_tokens.shape}, Valid tags: {valid_tags.shape}")
        print(f"Test tokens: {test_tokens.shape}, Test tags: {test_tags.shape}")
        
        # Print first few embeddings dictionary keys for debugging
        print("\nSample of embedding dictionary keys:")
        print(list(embeddings_dict.keys())[:5])
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    main()


