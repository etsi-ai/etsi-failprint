import pandas as pd
from etsi.failprint import analyze_nlp

def run_local_nlp_example():
    print("--- Running NLP Local Failure Example ---")
    
    # 1. Create synthetic data (No internet required)
    data = [
        ("The product works perfectly, very happy.", 0),
        ("Excellent customer support and fast delivery.", 0),
        ("User interface is clean and intuitive.", 0),
        ("I love using this app every day.", 0),
        ("Great value for money, highly recommended.", 0),
        
        ("This is the worst purchase I've ever made.", 1),
        ("Completely broken and useless after one day.", 1),
        ("Do not buy this, it is a total scam.", 1),
        ("Crash every time I open settings. Terrible.", 1),
        ("Refund refused. I am very angry.", 1),
        
        ("It's okay, but could be better.", 0),
        ("I am waiting for the update.", 0),
    ]
    
    df = pd.DataFrame(data, columns=['text', 'label'])
    
    # 2. Simulate Predictions
    y_true = df['label']
    y_pred = pd.Series([0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0], name="prediction")
    
    print(f"Data shape: {len(df)} samples")
    print("Simulating a model that fails to detect Negative sentiment...")
    
    # 3. Run failprint NLP analysis
    report = analyze_nlp(
        df['text'].tolist(),
        y_true,
        y_pred,
        output="markdown"
    )
    
    print("\nanalysis complete.")
    print(report)

if __name__ == "__main__":
    run_local_nlp_example()