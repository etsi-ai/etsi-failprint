import os
from sklearn.feature_extraction.text import TfidfVectorizer

class NlpReportWriter:
    """
    Writes reports specifically for NLP failure analysis.
    """
    def __init__(self, clustered_failures, output, log_path, total, failures, timestamp):
        self.clustered_failures = clustered_failures
        self.output = output
        self.log_path = log_path
        self.total = total
        self.failures = failures
        self.timestamp = timestamp

        # Ensure directories exist
        os.makedirs("reports", exist_ok=True)
        if not os.path.exists("failprint.log"):
            open("failprint.log", "w").close()

    def generate_markdown(self):
        """Generates the markdown report for NLP failures."""
        md_parts = [
            f"# failprint NLP Report",
            f"- Timestamp: {self.timestamp}",
            f"- Total Samples: {self.total}",
            f"- Failures: {self.failures} ({(self.failures / self.total) * 100:.2f}%)"
        ]

        if self.clustered_failures is None or self.clustered_failures.empty:
            md_parts.append("\n## No failure clusters found.")
            return "\n".join(md_parts)
            
        md_parts.append("\n## Failure Pattern Clusters")
        
        unique_clusters = sorted(self.clustered_failures['cluster'].unique())

        for cluster_id in unique_clusters:
            cluster_df = self.clustered_failures[self.clustered_failures['cluster'] == cluster_id]
            texts_in_cluster = cluster_df['text'].tolist()

            if cluster_id == -1:
                md_parts.append("\n---")
                md_parts.append("### Unique Failures (Noise)")
                md_parts.append(f"Found {len(texts_in_cluster)} unique failure(s) that don't fit a larger pattern.")
                for text in texts_in_cluster:
                    md_parts.append(f"- `{text}`")
                continue

            try:
                vectorizer = TfidfVectorizer(stop_words='english', max_features=3)
                vectorizer.fit(texts_in_cluster)
                keywords = vectorizer.get_feature_names_out()
            except ValueError:
                keywords = ["not enough data"]

            md_parts.append("\n---")
            md_parts.append(f"### Pattern {cluster_id}")
            md_parts.append(f"**Failures in this group:** {len(texts_in_cluster)}")
            md_parts.append(f"**Key Words:** `{', '.join(keywords)}`")
            md_parts.append("**Example Failures:**")
            for text in texts_in_cluster[:3]:
                md_parts.append(f"- `{text}`")

        return "\n".join(md_parts)

    def write(self):
        """Writes the report to a file and returns the markdown string."""
        markdown = self.generate_markdown()
        report_path = "reports/failprint_nlp_report.md"
        
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(markdown + "\n\n")
        
        with open(self.log_path, "a", encoding="utf-8") as log:
            log.write(f"[{self.timestamp}] NLP Failures: {self.failures}/{self.total}\n")
            
        return markdown

class ReportWriter:
    def __init__(self, segments, drift_map, clustered_segments,
                 output, log_path, total, failures, timestamp):
        self.segments = segments
        self.drift_map = drift_map
        self.clusters = clustered_segments
        self.output = output
        self.log_path = log_path
        self.total = total
        self.failures = failures
        self.timestamp = timestamp

        os.makedirs("reports", exist_ok=True)
        if not os.path.exists("failprint.log"):
            open("failprint.log", "w").close()
            print("[failprint] Created failprint.log")
        if not os.path.exists("reports/failprint_report.md"):
            open("reports/failprint_report.md", "w").close()
            print("[failprint] Created reports/failprint_report.md")

    def generate_markdown(self):
        md = [f"# failprint Report",
              f"- Timestamp: {self.timestamp}",
              f"- Total Samples: {self.total}",
              f"- Failures: {self.failures} ({(self.failures/self.total)*100:.2f}%)",
              "\n## Contributing Feature Segments"]

        for feat, vals in self.segments.items():
            md.append(f"**{feat}**:")
            for val, fail_pct, delta in vals:
                md.append(f"- `{val}` → {fail_pct*100:.1f}% in failures (Δ +{delta*100:.1f}%)")
        return "\n".join(md)

    def write(self):
        markdown = self.generate_markdown()
        with open("reports/failprint_report.md", "w", encoding="utf-8") as f:
            f.write(markdown + "\n\n")
        with open(self.log_path, "a", encoding="utf-8") as log:
            log.write(f"[{self.timestamp}] Failures: {self.failures}/{self.total}\n")
        return markdown
