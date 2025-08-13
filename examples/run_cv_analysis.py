# examples/run_cv_analysis.py
try:
    from etsi.failprint import analyze_cv
except ImportError:
    import sys
    import os
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from etsi.failprint import analyze_cv

def run_example():
    """
    Demonstrates how to use analyze_cv with a sample set of images.
    """
    print("--- Running Computer Vision (CV) Failure Analysis Example ---")

    image_paths = [
        "examples/test_images/cat_1.png",
        "examples/test_images/cat_2_dark.png",    # This will be a failure
        "examples/test_images/dog_1.png",
        "examples/test_images/dog_2_blurry.png",  # This will be a failure
        "examples/test_images/cat_3.png"      # Assuming you add another cat photo
    ]
    
    y_true = [0, 0, 1, 1, 0]

    # 2. Simulate model predictions, with two intentional errors
    # The model correctly identifies the clear cat and dog, but fails on the dark and blurry ones.
    y_pred = [0, 1, 1, 0, 0] # Fails on cat_2_dark (predicts 1) and dog_2_blurry (predicts 0)

    print(f"\nAnalyzing {len(image_paths)} images with {sum(p != t for p, t in zip(y_pred, y_true))} failures...")
    
    # 3. Run the failprint analysis
    report = analyze_cv(
        image_paths=image_paths,
        y_true=y_true,
        y_pred=y_pred,
        model_name="animal_classifier_v1",
        cluster_failures=True, # Set to True to generate visual clusters
    )

    # 4. Print the final report
    print("\n--- failprint CV Report ---")
    print(report)
    print("\nReport saved to reports/failprint_cv_report.md")
    print("Check the reports/ folder for the image cluster collages!")

if __name__ == "__main__":
    run_example()