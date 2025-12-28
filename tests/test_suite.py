import unittest
import pandas as pd
import numpy as np
import os
from etsi.failprint import analyze

class TestFailprintCore(unittest.TestCase):

    def setUp(self):
        # UPDATED: Use only numeric features to pass validation
        self.X = pd.DataFrame({
            "age": [25, 30, 35, 40, 45, 22, 28, 50],
            "salary": [50000, 60000, 65000, 80000, 90000, 45000, 55000, 120000],
            "tenure": [1, 2, 3, 4, 5, 1, 2, 10] 
        })
        
        # Create a pattern: High salary (>85000) always fails
        self.y_true = pd.Series([0, 0, 1, 1, 0, 0, 1, 1])
        self.y_pred = pd.Series([0, 0, 1, 1, 1, 0, 1, 0]) 

    def test_analyze_execution(self):
        """Test if analyze runs without error and produces files."""
        # Clean up previous runs
        if os.path.exists("reports/failprint_report.md"):
            os.remove("reports/failprint_report.md")
            
        report = analyze(
            self.X, 
            self.y_true, 
            self.y_pred, 
            output="markdown", 
            cluster=True
        )
        
        # Check output type
        self.assertIsInstance(report, str)
        # If validation fails, report will start with "Error:", so we check for success header
        self.assertIn("failprint Report", report)
        
        # Check file creation
        self.assertTrue(os.path.exists("reports/failprint_report.md"))
        self.assertTrue(os.path.exists("failprint.log"))

    def test_input_validation(self):
        """Test if it catches mismatched lengths gracefully."""
        y_short = self.y_true.iloc[:-1]
        
        # The analyze function now returns an error string instead of raising an exception
        result = analyze(self.X, y_short, self.y_pred)
        
        self.assertIsInstance(result, str)
        self.assertTrue(result.startswith("Error:"), f"Expected error message, got: {result[:50]}")

if __name__ == '__main__':
    unittest.main()