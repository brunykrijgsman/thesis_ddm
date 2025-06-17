import pickle

def save_training_history(history, path):
    """Save training history as pickle."""
    
    # Save as pickle - include seed in filename
    with open(path, 'wb') as f:
        pickle.dump(history.history, f)
    
    return path

def load_training_history(path):
    """Load training history from a pickle file."""
    with open(path, 'rb') as f:
        return pickle.load(f) 