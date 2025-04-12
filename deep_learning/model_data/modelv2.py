import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import re
import pickle
import collections
import sqlite3
import pandas as pd 
import os
from tqdm import tqdm
import random
import json

class EnhancedSQLTokenizer:
    def __init__(self):
        # More comprehensive vocabulary
        self.vocab = {
            '<PAD>': 0,
            '<UNK>': 1,
            '<START>': 2,
            '<END>': 3
        }
        conn = sqlite3.connect("new_courses.db")
        cursor = conn.cursor()

        # Get all table names from the database
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]
        
        # Get column names for each table
        all_columns = []
        for table in tables:
            cursor.execute(f"PRAGMA table_info({table});")
            table_columns = [row[1] for row in cursor.fetchall()]
            all_columns.extend(table_columns)

        # Expanded token categories with more SQL-specific tokens
        predefined_token_categories = {
            'sql_keywords': [
                'SELECT', 'FROM', 'WHERE', 'AND', 'OR', 'NOT', 'IN', 'LIKE',
                'IS', 'NULL', 'BY', 'ORDER', 'HAVING', 'LIMIT', '*'
            ],
            'sql_operators': [
                '=', '!=', '<', '>', '<=', '>=', '<>', 'LIKE', '%',"'",
            ],
            'aggregations': [
                'COUNT', 'MAX', 'DISTINCT'
            ],
            'table_terms': tables,
            'column_terms': all_columns,
            'instructors': [f'{row[0]}' for row in cursor.execute("SELECT DISTINCT instructor FROM complete_courses WHERE instructor IS NOT NULL AND instructor != ''").fetchall()],
            'dept_codes': [f'{row[0]}' for row in cursor.execute("SELECT DISTINCT department_code FROM complete_courses WHERE department_code IS NOT NULL AND department_code != ''").fetchall()],
            'course_codes': [f'{row[0]}' for row in cursor.execute("SELECT DISTINCT course_code FROM complete_courses WHERE course_code IS NOT NULL AND course_code != ''").fetchall()],
            'faculties': [f'{row[0]}' for row in cursor.execute("SELECT DISTINCT faculty FROM complete_courses WHERE faculty IS NOT NULL AND faculty != ''").fetchall()],
            'dept_names': [f'{row[0]}' for row in cursor.execute("SELECT DISTINCT department_name FROM complete_courses WHERE department_name IS NOT NULL AND department_name != ''").fetchall()],
            'numeric_qualifiers': [
                '1', '2', '3', '4', '6.0'
            ],
            'special_conditions': [
                'no', 'with', 'without', 'all'
            ],
            'brackets': ['[',']']
        }
        
        # Populate vocabulary
        for category, tokens in predefined_token_categories.items():
            for token in tokens:
                if token is not None:  # Skip None values
                    token_str = str(token)
                    if token_str not in self.vocab:
                        self.vocab[token_str] = len(self.vocab)
        
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
            token_index = self.vocab.get(token, self.vocab['<UNK>'])
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
        for idx in indices:
            token = self.reverse_vocab.get(idx, '<UNK>')
            if token in ['<START>', '<END>', '<PAD>']:
                continue
            decoded_tokens.append(token)
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
        
        # Create checkpoint directory if it doesn't exist
        self.checkpoint_dir = "checkpoints"
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
            
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
    
    def save_checkpoint(self, model, optimizer, epoch, loss, filename):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }
        torch.save(checkpoint, filename)
        print(f"Checkpoint saved: {filename}")
    
    def load_checkpoint(self, model, optimizer, filename):
        """Load model checkpoint"""
        if os.path.isfile(filename):
            checkpoint = torch.load(filename)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch = checkpoint['epoch']
            loss = checkpoint['loss']
            print(f"Checkpoint loaded: {filename}")
            return model, optimizer, epoch, loss
        else:
            print(f"No checkpoint found at {filename}")
            return model, optimizer, 0, float('inf')
    
    def train(self, 
        embedding_dim=128, 
        hidden_dim=256, 
        epochs=5, 
        learning_rate=0.005,
        batch_size=32,
        checkpoint_interval=50,  # Save checkpoint every n epochs
        resume_training=False):  # Option to resume from checkpoint
    
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
    
    # Loss and optimizer
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.5, 
            patience=100
    )
    
    # Resume from checkpoint if requested
        start_epoch = 0
        best_loss = float('inf')
        if resume_training:
            latest_checkpoint = self._get_latest_checkpoint()
            if latest_checkpoint:
                try:
                # First try loading with strict=True
                    model, optimizer, start_epoch, best_loss = self.load_checkpoint(
                        model, optimizer, latest_checkpoint
                )
                except RuntimeError as e:
                # If there's a size mismatch due to vocabulary expansion
                    if "size mismatch" in str(e):
                        print("Detected vocabulary size change. Loading with strict=False and extending embeddings...")
                    # Load checkpoint with strict=False
                        checkpoint = torch.load(latest_checkpoint)
                    
                    # Get the old vocabulary size from the checkpoint
                        old_vocab_size = checkpoint['model_state_dict']['encoder.embedding.weight'].size(0)
                        print(f"Old vocabulary size: {old_vocab_size}, New vocabulary size: {input_vocab_size}")
                    
                    # Load state dict with strict=False
                        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                    
                    # Extend the embeddings
                    
                    # Load optimizer state
                        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                        start_epoch = checkpoint['epoch']
                        best_loss = checkpoint['loss']
                        print(f"Checkpoint loaded and embeddings extended: {latest_checkpoint}")
                    else:
                    # Re-raise if it's not a size mismatch issue
                        raise
    
    # Training loop with progress bars
        patience = 500
        trigger_times = 0
        iterations_per_epoch = 20
        try:
        # Outer progress bar for epochs
            epoch_bar = tqdm(range(start_epoch, epochs), desc="Training Progress", position=0)
        
            for epoch in epoch_bar:
            # Zero gradients
                optimizer.zero_grad()
                iter_bar = tqdm(range(iterations_per_epoch), desc=f"Epoch {epoch+1}/{epochs}", position=1, leave=False)

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
            
            # Learning rate scheduling
                scheduler.step(loss)
            
            # Decode the output (predicted tokens) for display
                output_ids = outputs.argmax(dim=2)  # Get the predicted token IDs
                decoded_sql = [self.sql_tokenizer.decode(output_ids[i].tolist()) for i in range(output_ids.size(0))]

            # Print decoded results
                print(f"\nDecoded SQL Outputs at Epoch {epoch+1}:")
                sampled_decoded_sql = random.sample(decoded_sql, min(10, len(decoded_sql)))  # Randomly select up to 10 samples
                for i, decoded in enumerate(sampled_decoded_sql):
                    print(f"Sample {i+1}: {decoded}")
            
            # Update progress bar description
                epoch_bar.set_description(
                    f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}"
            )
            
            # Save checkpoint at regular intervals
                if (epoch + 1) % checkpoint_interval == 0:
                    self.save_checkpoint(
                        model, 
                        optimizer, 
                        epoch + 1, 
                        loss.item(), 
                        f"{self.checkpoint_dir}/checkpoint_epoch_{epoch+1}.pth"
              )
            
            # Save best model
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    trigger_times = 0
                    self.save_checkpoint(
                        model, 
                        optimizer,
                        epoch + 1,
                        loss.item(),
                        f"{self.checkpoint_dir}/best_model.pth"
              )
                else:
                    trigger_times += 1
            
                if trigger_times >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        except KeyboardInterrupt:
            print("Training interrupted. Saving current checkpoint.")
            self.save_checkpoint(
                model, 
                optimizer, 
                epoch + 1, 
                loss.item(), 
                f"{self.checkpoint_dir}/interrupt_checkpoint.pth"
        )
            
        self.analyze_token_generation(model, self.nl_sequences, self.sql_sequences)
        return model, self.nl_tokenizer, self.sql_tokenizer
    
    def _get_latest_checkpoint(self):
        """Get the latest checkpoint file"""
        checkpoints = [f for f in os.listdir(self.checkpoint_dir) 
                      if f.startswith("checkpoint_epoch_") and f.endswith(".pth")]
        if not checkpoints:
            return None
            
        # Extract epoch numbers and find the latest
        epochs = [int(re.search(r'checkpoint_epoch_(\d+)\.pth', ckpt).group(1)) 
                 for ckpt in checkpoints]
        latest_epoch = max(epochs)
        return f"{self.checkpoint_dir}/checkpoint_epoch_{latest_epoch}.pth"
    
    def analyze_token_generation(self, model, nl_sequences, sql_sequences):
        """Analyze how the model generates tokens"""
        model.eval()
        with torch.no_grad():
            # Generate translations for input sequences
            outputs = model(nl_sequences, sql_sequences)
            
            # Get predicted tokens
            predicted_tokens = outputs.argmax(dim=2)
             # Debugging: Output token IDs and decoded SQL
            output_ids = [tensor.item() for tensor in predicted_tokens[0]]  # Taking the first batch item for now
            print("Output token IDs:", output_ids)  # List of token indices
            decoded_sql = self.sql_tokenizer.decode(output_ids)  # Decode the output token indices to SQL query
            print("Decoded SQL:", decoded_sql)  # Print the decoded SQL query

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
                
                # Get most likely token
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
        
def load_spider_dataset():
    """Load and preprocess the Spider dataset, adapting it to the course database schema"""
    import json
    from tqdm import tqdm
    
    # Paths to Spider files - update these to your actual paths
    train_path = "spider/train_spider.json"
    dev_path = "spider/dev.json"
    
    nl_queries = []
    sql_queries = []
    
    try:
        # Load training data
        with open(train_path, 'r') as f:
            train_data = json.load(f)
        
        # Load dev data
        with open(dev_path, 'r') as f:
            dev_data = json.load(f)
            
        # Combine data
        all_data = train_data + dev_data
        
        # Define schema mappings
        table_map = {
            "student": "complete_courses",
            "course": "complete_courses",
            "professor": "complete_courses",
            "department": "complete_courses",
            "program": "complete_courses",
            "instructor": "complete_courses"
        }
        
        column_map = {
            "course_id": "course_code",
            "title": "course_name",
            "credits": "units",
            "level": "year",
            "dept_name": "department_name",
            "dept_id": "department_code",
            "professor_name": "instructor",
            "prereq": "prereq_codes",
            "description": "description",
            "requirements": "requirements",
            "college": "faculty",
            "outcomes": "outcomes",
            "coreq": "corequisites",
            "exclusions": "exclusions",
            "recommended": "recommended",
            "tags": "keywords"
        }
        
        processed = 0
        adapted = 0
        
        for item in tqdm(all_data, desc="Adapting Spider data to course schema"):
            processed += 1
            
            # Skip queries for tables that don't match our schema
            if not any(table in item['query'].lower() for table in table_map.keys()):
                continue
                
            # Get the original query
            nl_query = item['question']
            sql_query = item['query']
            
            # Replace table and column names
            for old_table, new_table in table_map.items():
                # Handle SQL query
                # Only match standalone table names with spaces/punctuation around them
                sql_query = re.sub(rf'\b{old_table}\b', new_table, sql_query, flags=re.IGNORECASE)
                
                # Handle NL query - more flexible matching for natural language
                nl_query = re.sub(rf'\b{old_table}s?\b', new_table, nl_query, flags=re.IGNORECASE)
            
            for old_col, new_col in column_map.items():
                # Handle SQL query - only match standalone column names
                sql_query = re.sub(rf'\b{old_col}\b', new_col, sql_query, flags=re.IGNORECASE)
                
                # Handle NL query
                nl_query = re.sub(rf'\b{old_col}s?\b', new_col, nl_query, flags=re.IGNORECASE)
            
            # Replace any table name other than complete_courses
            # This handles tables in Spider that we don't have explicit mappings for
            sql_query = re.sub(r'FROM\s+([a-zA-Z_][a-zA-Z0-9_]*)\s+', 'FROM complete_courses ', sql_query, flags=re.IGNORECASE)
            sql_query = re.sub(r'JOIN\s+([a-zA-Z_][a-zA-Z0-9_]*)\s+', 'JOIN complete_courses ', sql_query, flags=re.IGNORECASE)
            
            # Add adapted queries to our lists
            nl_queries.append(nl_query)
            sql_queries.append(sql_query)
            adapted += 1
        
        print(f"Processed {processed} Spider queries, adapted {adapted} to course schema")
        
    except FileNotFoundError as e:
        print(f"Error loading Spider dataset: {e}")
        print("If you don't have the Spider dataset, download it from https://yale-lily.github.io/spider")
        return [], []
        
    return nl_queries, sql_queries

def main():
    # Training data with more diverse examples
    print("Loading course-specific training data...")
    train_df = pd.read_csv('course_query_training_data.csv')
    course_sql_queries = train_df['sql_query'].tolist()
    course_nl_queries = train_df['natural_language_query'].tolist()
    
    # Load Spider data adapted to our schema
    print("Loading and adapting Spider dataset...")
    spider_nl, spider_sql = load_spider_dataset()
    
    # Combine datasets
    if spider_nl and spider_sql:
        print(f"Original course dataset: {len(course_nl_queries)} examples")
        print(f"Adapted Spider dataset: {len(spider_nl)} examples")
        
        # Option to balance the datasets if Spider is much larger
        if len(spider_nl) > 3 * len(course_nl_queries):
            sample_size = 3 * len(course_nl_queries)  # Use at most 3x our original data
            sampled_indices = random.sample(range(len(spider_nl)), min(sample_size, len(spider_nl)))
            spider_nl = [spider_nl[i] for i in sampled_indices]
            spider_sql = [spider_sql[i] for i in sampled_indices]
            print(f"Sampled Spider dataset: {len(spider_nl)} examples")
        
        # Combine datasets
        nl_queries = course_nl_queries + spider_nl
        sql_queries = course_sql_queries + spider_sql
        print(f"Combined dataset: {len(nl_queries)} examples")
        
        # Shuffle the combined dataset
        combined = list(zip(nl_queries, sql_queries))
        random.shuffle(combined)
        nl_queries, sql_queries = zip(*combined)
    else:
        print("Using only course-specific data")
        nl_queries = course_nl_queries
        sql_queries = course_sql_queries

    # Create checkpoint directory
    checkpoint_dir = "checkpoints"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # Train the model with checkpoint saving
    trainer = NLToSQLTrainer(nl_queries, sql_queries)
    
    # Analyze input data
    trainer.analyze_input_data(nl_queries, sql_queries)
    
    # Check if we should resume training
    resume_training = False
    if os.path.exists(f"{checkpoint_dir}/interrupt_checkpoint.pth") or \
       os.path.exists(f"{checkpoint_dir}/best_model.pth"):
        resume = input("Found existing checkpoints. Resume training? (y/n): ")
        resume_training = resume.lower() == 'y'
    
    # Handle case where model vocabulary has changed due to Spider data
    if resume_training and os.path.exists(f"{checkpoint_dir}/checkpoint_epoch_250.pth"):
        # We'll use the extend_embeddings function later in the train method
        print("Note: Using Spider dataset may require extending model embeddings if vocabulary size has changed")
    
    # Set lower memory requirements
    embedding_dim = 64  # Reduced from 128
    hidden_dim = 128    # Reduced from 256
    batch_size = 16     # Reduced batch size for lower memory usage
    
    model, nl_tokenizer, sql_tokenizer = trainer.train(
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        epochs=450,
        batch_size=batch_size,
        checkpoint_interval=25,  # Save every 25 epochs
        resume_training=resume_training
    )

    # Create inference object
    inference = NLToSQLInference(model, nl_tokenizer, sql_tokenizer)

    # Test inference
    test_queries = [
        "show me all physics courses in first year",
        "find courses with no prerequisites",
        "list all third year psychology courses",
        "courses in the computer science department",
        # Add some queries inspired by Spider dataset
        "what is the average number of units for computer science courses?",
        "find courses with the highest units in the engineering faculty",
        "show me courses taught by Professor Smith that have programming in the description"
    ]

    for query in test_queries:
        print(f"\nNatural Language Query: {query}")
        sql_query = inference.translate(query)
        print(f"Generated SQL Query: {sql_query}")

    # Save final model and tokenizers
    torch.save(model.state_dict(), 'nl_to_sql_model.pth')
    with open('nl_tokenizer.pkl', 'wb') as f:
        pickle.dump(nl_tokenizer, f)
    with open('sql_tokenizer.pkl', 'wb') as f:
        pickle.dump(sql_tokenizer, f)
    
    print("\nTraining complete! Model and tokenizers saved.")
    print("You can load them with:")
    print("  - Model: torch.load('nl_to_sql_model.pth')")
    print("  - NL Tokenizer: pickle.load(open('nl_tokenizer.pkl', 'rb'))")
    print("  - SQL Tokenizer: pickle.load(open('sql_tokenizer.pkl', 'rb'))")

if __name__ == '__main__':
    main()