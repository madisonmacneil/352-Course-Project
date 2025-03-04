import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import re
import pickle
import collections

class EnhancedSQLTokenizer:
    def __init__(self):
        # More comprehensive vocabulary
        self.vocab = {
            '<PAD>': 0,
            '<UNK>': 1,
            '<START>': 2,
            '<END>': 3
        }
        
        # Expanded token categories with more SQL-specific tokens
        predefined_token_categories = {
            'sql_keywords': [
                'SELECT', 'FROM', 'WHERE', 'AND', 'OR', 'NOT', 'IN', 'LIKE',
                'IS', 'NULL', 'GROUP', 'BY', 'ORDER', 'HAVING', 'LIMIT', '*'
            ],
            'sql_operators': [
                '=', '!=', '<', '>', '<=', '>=', '<>', 'LIKE'
            ],
            'aggregations': [
                'COUNT', 'SUM', 'AVG', 'MAX', 'MIN', 'DISTINCT'
            ],
            'table_terms': [
                'courses', 'students', 'enrollment', 'department'
            ],
            'column_terms': [
                'course_name', 'credits', 'semester', 'year', 'term', 
                'prerequisites', 'department'
            ],
            'subject_domains': [
                'physics', 'psychology', 'computer', 'science', 'ethics',
                'computer science'
            ],
            'numeric_qualifiers': [
                'first', 'second', 'third', 'fourth',
                '1st', '2nd', '3rd', '4th'
            ],
            'special_conditions': [
                'no', 'with', 'without', 'all'
            ]
        }
        
        # Populate vocabulary
        for category, tokens in predefined_token_categories.items():
            for token in tokens:
                lower_token = token.lower()
                if lower_token not in self.vocab:
                    self.vocab[lower_token] = len(self.vocab)
        
        # Create reverse vocabulary
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        
        print("Vocabulary Size:", len(self.vocab))
        print("Top 20 Tokens:", list(self.vocab.keys())[:20])
    
    def tokenize(self, query, max_length=75):
        # More sophisticated tokenization
        pattern = r'(\b[A-Za-z_]+\b|[*=<>]+|\d+|\(|\))'
        
        # Tokenize
        tokens = re.findall(pattern, query, re.IGNORECASE)
        
        # Convert to lowercase
        tokens = [token.lower() for token in tokens]
        
        # Convert to indices
        token_indices = [self.vocab.get('<START>')] # Start token
        
        for token in tokens:
            # Prioritize exact match, then partial match
            token_index = (
                self.vocab.get(token) or 
                next((self.vocab.get(t) for t in self.vocab if token in t), 
                     self.vocab['<UNK>'])
            )
            token_indices.append(token_index)
        
        # Add end token
        token_indices.append(self.vocab['<END>'])
        
        # Pad or truncate
        if len(token_indices) > max_length:
            token_indices = token_indices[:max_length]
        else:
            token_indices = token_indices + [self.vocab['<PAD>']] * (max_length - len(token_indices))
        
        return torch.tensor(token_indices, dtype=torch.long)
    
    def decode(self, indices):
        decoded_tokens = []
        seen_tokens = set()
        
        for idx in indices:
            if idx >= 4:  # Skip special tokens
                token = self.reverse_vocab.get(idx, '<UNK>')
                if token not in seen_tokens and token != '<unk>':
                    decoded_tokens.append(token)
                    seen_tokens.add(token)
        
        return ' '.join(decoded_tokens)


class NLToSQLEncoder(nn.Module):
    def __init__(self, input_vocab_size, embedding_dim=128, hidden_dim=256, num_layers=2):
        super(NLToSQLEncoder, self).__init__()
        # Embedding layer with larger dimension
        self.embedding = nn.Embedding(input_vocab_size, embedding_dim, padding_idx=0)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.3)
        
        # LSTM Encoder with more complex configuration
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim // 2,  # Bidirectional adjustment
            num_layers, 
            batch_first=True, 
            bidirectional=True,
            dropout=0.3
        )
    
    
    def forward(self, x):
        # Embed input with dropout
        embedded = self.dropout(self.embedding(x))
        
        # Pass through LSTM
        outputs, (hidden, cell) = self.lstm(embedded)
        
        # Reshape hidden and cell states
        batch_size = x.size(0)
        num_layers = hidden.size(0) // 2  # Because bidirectional
        hidden_size = hidden.size(2)
        
        # Combine forward and backward hidden states
        hidden = hidden.view(num_layers, 2, batch_size, hidden_size)
        hidden = hidden.transpose(1, 2).contiguous()
        hidden = hidden.view(num_layers, batch_size, -1)
        
        cell = cell.view(num_layers, 2, batch_size, hidden_size)
        cell = cell.transpose(1, 2).contiguous()
        cell = cell.view(num_layers, batch_size, -1)
        
        return hidden, cell

class NLToSQLDecoder(nn.Module):
    def __init__(self, output_vocab_size, embedding_dim=128, hidden_dim=256, num_layers=2):
        super(NLToSQLDecoder, self).__init__()
        # Embedding layer
        self.embedding = nn.Embedding(output_vocab_size, embedding_dim, padding_idx=0)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.3)
        
        # LSTM Decoder
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True,
            dropout=0.3
        )
        
        # Output layer with attention-like mechanism
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_vocab_size)
        )
    
    def forward(self, x, hidden, cell):
        # Embed input with dropout
        embedded = self.dropout(self.embedding(x))
        
        # Pass through LSTM
        outputs, (new_hidden, new_cell) = self.lstm(embedded, (hidden, cell))
        
        # Predict next token
        predictions = self.fc(outputs)
        
        return predictions.squeeze(1), new_hidden, new_cell

class NLToSQLSeq2Seq(nn.Module):
    def __init__(self, input_vocab_size, output_vocab_size, 
                 embedding_dim=128, hidden_dim=256, num_layers=2):
        super(NLToSQLSeq2Seq, self).__init__()
        # Encoder and Decoder
        self.encoder = NLToSQLEncoder(
            input_vocab_size, 
            embedding_dim, 
            hidden_dim, 
            num_layers
        )
        self.decoder = NLToSQLDecoder(
            output_vocab_size, 
            embedding_dim, 
            hidden_dim, 
            num_layers
        )
    
    def forward(self, input_seq, target_seq):
        batch_size = input_seq.size(0)
        target_len = target_seq.size(1)
        
        # Encode input sequence
        hidden, cell = self.encoder(input_seq)
        
        # Prepare output tensor
        outputs = torch.zeros(
            batch_size, 
            target_len, 
            self.decoder.fc[-1].out_features
        ).to(input_seq.device)
        
        # First decoder input
        decoder_input = target_seq[:, 0].unsqueeze(1)
        
        # Autoregressive decoding
        for t in range(1, target_len):
            # Decode
            output, hidden, cell = self.decoder(
                decoder_input, hidden, cell
            )
            
            # Store predictions
            outputs[:, t] = output
            
            # Use actual next token (teacher forcing)
            decoder_input = target_seq[:, t].unsqueeze(1)
        
        return outputs

class NLToSQLTrainer:
    def __init__(self, nl_queries, sql_queries):
        # Create enhanced data augmentation
        nl_queries, sql_queries = self.advanced_data_augmentation(nl_queries, sql_queries)
        
        # Create tokenizers
        self.nl_tokenizer = EnhancedSQLTokenizer()
        self.sql_tokenizer = EnhancedSQLTokenizer()
        
        # Encode sequences
        self.nl_sequences = torch.stack([
            self.nl_tokenizer.tokenize(query) 
            for query in nl_queries
        ])
        
        self.sql_sequences = torch.stack([
            self.sql_tokenizer.tokenize(query) 
            for query in sql_queries
        ])
    def analyze_input_data(self, nl_queries, sql_queries):
        """Comprehensive analysis of input data"""
        print("\n--- Input Data Analysis ---")
        
        # Token frequency in natural language queries
        nl_tokens = [token.lower() 
                     for query in nl_queries 
                     for token in re.findall(r'\b[A-Za-z_]+\b', query)]
        nl_token_freq = collections.Counter(nl_tokens)
        
        # Token frequency in SQL queries
        sql_tokens = [token.lower() 
                      for query in sql_queries 
                      for token in re.findall(r'\b[A-Za-z_]+\b', query)]
        sql_token_freq = collections.Counter(sql_tokens)
        
        print("Top 10 NL Tokens:", nl_token_freq.most_common(10))
        print("Top 10 SQL Tokens:", sql_token_freq.most_common(10))
        
        # Analyze query complexity
        print(f"Average NL Query Length: {np.mean([len(q.split()) for q in nl_queries]):.2f}")
        print(f"Average SQL Query Length: {np.mean([len(q.split()) for q in sql_queries]):.2f}")

    
    def advanced_data_augmentation(self, nl_queries, sql_queries):
        """
        More sophisticated data augmentation
        """
        augmented_nl = []
        augmented_sql = []
        
        # Synonym and rephrasing dictionaries
        nl_synonyms = {
            'courses': ['classes', 'course', 'class', 'subjects'],
            'physics': ['physical sciences', 'physics department', 'physical science'],
            'psychology': ['psych', 'psychology department', 'mental studies'],
            'first': ['1st', 'initial', 'freshman'],
            'third': ['3rd', 'final', 'senior'],
            'winter': ['winter term', 'winter semester'],
            'fall': ['fall term', 'fall semester']
        }
        
        # Original data first
        for nl, sql in zip(nl_queries, sql_queries):
            augmented_nl.append(nl)
            augmented_sql.append(sql)
            
            # Generate variations
            for word, replacements in nl_synonyms.items():
                for replacement in replacements:
                    # Create new natural language query
                    new_nl = nl.replace(word, replacement)
                    
                    # Only add if it's a meaningful variation
                    if new_nl != nl:
                        augmented_nl.append(new_nl)
                        augmented_sql.append(sql)
        
        return augmented_nl, augmented_sql
    
    def train(self, 
          embedding_dim=128, 
          hidden_dim=256, 
          epochs=500, 
          learning_rate=0.005):
    # Model parameters
        input_vocab_size = len(self.nl_tokenizer.vocab)
        output_vocab_size = len(self.sql_tokenizer.vocab)
    
    # Initialize model
        model = NLToSQLSeq2Seq(
            input_vocab_size, 
            output_vocab_size, 
            embedding_dim, 
            hidden_dim
        )
    
    # Loss and optimizer with learning rate scheduling
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.5, 
            patience=100
        )
    
    # Training loop with early stopping
        best_loss = float('inf')
        patience = 500
        trigger_times = 0
    
        for epoch in range(epochs):
        # Zero gradients
            optimizer.zero_grad()
        
        # Forward pass
            outputs = model(self.nl_sequences, self.sql_sequences)

        # Compute loss
            loss = criterion(
                outputs.reshape(-1, output_vocab_size), 
                self.sql_sequences.reshape(-1)
        )
        
        # Backward pass
            loss.backward()
        
        # Optimize
            optimizer.step()
        
        # Learning rate scheduling (remove epoch parameter)
            scheduler.step(loss)
        
        # Early stopping
            if loss.item() < best_loss:
                best_loss = loss.item()
                trigger_times = 0
            # Optional: Save best model
            # torch.save(model.state_dict(), 'best_model.pth')
            else:
                trigger_times += 1
        
        # Verbose logging
            if epoch % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
        
        # Early stopping condition
            if trigger_times >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

        self.analyze_token_generation(model, self.nl_sequences, self.sql_sequences)
        return model, self.nl_tokenizer, self.sql_tokenizer

    def analyze_token_generation(self, model, nl_sequences, sql_sequences):
        """Analyze how the model generates tokens"""
        model.eval()
        with torch.no_grad():
            # Generate translations for input sequences
            outputs = model(nl_sequences, sql_sequences)
            
            # Get predicted tokens
            predicted_tokens = outputs.argmax(dim=2)
            
            # Analyze token distribution
            token_counts = {}
            for batch_tokens in predicted_tokens:
                for token in batch_tokens:
                    token = token.item()
                    token_counts[token] = token_counts.get(token, 0) + 1
            
            # Convert to percentages
            total_tokens = sum(token_counts.values())
            token_percentages = {
                self.sql_tokenizer.reverse_vocab.get(k, '<UNK>'): 
                (v / total_tokens) * 100 
                for k, v in token_counts.items()
            }
            
            # Sort and print
            sorted_tokens = sorted(
                token_percentages.items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            
            print("\n--- Token Generation Analysis ---")
            print("Top Token Generations:")
            for token, percentage in sorted_tokens[:10]:
                print(f"{token}: {percentage:.2f}%")
# Rest of the code remains the same as in the previous implementation
# (NLToSQLInference and main() functions)

class NLToSQLInference:
    def __init__(self, model, nl_tokenizer, sql_tokenizer):
        self.model = model
        self.nl_tokenizer = nl_tokenizer
        self.sql_tokenizer = sql_tokenizer
    
    def translate(self, nl_query, max_length=50):
        # Prepare input sequence
        input_seq = self.nl_tokenizer.tokenize(nl_query).unsqueeze(0)
        
        # Prepare initial target sequence
        target_seq = torch.zeros((1, max_length), dtype=torch.long)
        target_seq[0, 0] = self.sql_tokenizer.vocab['<START>']
        
        # Disable gradient computation
        with torch.no_grad():
            # Encode input sequence
            hidden, cell = self.model.encoder(input_seq)
            
            # Generate sequence
            decoded_tokens = []
            decoder_input = target_seq[0, 0].unsqueeze(0).unsqueeze(0)
            
            for _ in range(max_length - 1):
                # Decode
                output, hidden, cell = self.model.decoder(decoder_input, hidden, cell)
                
                # Get most likely token (fix dimensional issue)
                predicted_token = output.argmax(dim=-1).item()
                
                # Stop if end token or repeated token
                if (predicted_token == self.sql_tokenizer.vocab['<END>'] 
                    or predicted_token in decoded_tokens):
                    break
                
                decoded_tokens.append(predicted_token)
                
                # Update decoder input
                decoder_input = torch.tensor([[predicted_token]], dtype=torch.long)
            
            # Convert to SQL query
            sql_query = self.sql_tokenizer.decode(decoded_tokens)
        
        return sql_query


def main():
    # Training data with more diverse examples
    nl_queries = [
        "show me all physics courses in first year",
        "find courses in the winter term with no prerequisites",
        "list all third year psychology courses",
        "get third year courses with at least 6 credits",
        "find ethics courses with no prerequisites",
        "show me an elective without prerequisites",
        "courses in computer science department",
        "find fall semester courses",
        "list courses with more than 3 credits"
    ]

    sql_queries = [
        "SELECT * FROM courses WHERE department = 'Physics' AND year = 1",
        "SELECT * FROM courses WHERE term = 'Winter' AND (prerequisites IS NULL OR prerequisites = '')",
        "SELECT * FROM courses WHERE department = 'Psychology' AND year = 3",
        "SELECT * FROM courses WHERE year = 3 AND credits >= 6",
        "SELECT * FROM courses WHERE course_name LIKE '%ethics%' AND prerequisites IS NULL",
        "SELECT * FROM courses WHERE prerequisites IS NULL",
        "SELECT * FROM courses WHERE department = 'Computer Science'",
        "SELECT * FROM courses WHERE term = 'Fall'",
        "SELECT * FROM courses WHERE credits > 3"
    ]

    # Train the model
    trainer = NLToSQLTrainer(nl_queries, sql_queries)
    model, nl_tokenizer, sql_tokenizer = trainer.train()

    # Create inference object
    inference = NLToSQLInference(model, nl_tokenizer, sql_tokenizer)

    # Test inference
    test_queries = [
        "show me all physics courses in first year",
        "find courses in the winter term with no prerequisites",
        "list all third year psychology courses",
        "courses in computer science department"
    ]

    for query in test_queries:
        print(f"\nNatural Language Query: {query}")
        sql_query = inference.translate(query)
        print(f"Generated SQL Query: {sql_query}")

    # Optional: Save model and tokenizers
    torch.save(model.state_dict(), 'nl_to_sql_model.pth')
    with open('nl_tokenizer.pkl', 'wb') as f:
        pickle.dump(nl_tokenizer, f)
    with open('sql_tokenizer.pkl', 'wb') as f:
        pickle.dump(sql_tokenizer, f)

if __name__ == '__main__':
    main()