import os
from PIL import Image
import math
from sklearn.feature_extraction.text import TfidfVectorizer

class NlpReportWriter:
    """
    Writes reports specifically for NLP failure analysis, including
    feature-based segments and semantic clusters.
    """
    # --- MODIFIED: Add 'nlp_segments' to the constructor ---
    def __init__(self, clustered_failures, nlp_segments, output, log_path, total, failures, timestamp):
        self.clustered_failures = clustered_failures
        self.nlp_segments = nlp_segments  # Store the new segments
        self.output = output
        self.log_path = log_path
        self.total = total
        self.failures = failures
        self.timestamp = timestamp

        os.makedirs("reports", exist_ok=True)
        if not os.path.exists("failprint.log"):
            open("failprint.log", "w").close()

    def _generate_segment_markdown(self):
        if not self.nlp_segments:
            return []
            
        md_parts = ["\n## Failure Segments by Text Characteristics"]
        md_parts.append("Segments where a feature value is over-represented in failures compared to the overall dataset.")
        
        for feature, values in self.nlp_segments.items():
            pretty_feature = feature.replace('_', ' ').title()
            md_parts.append(f"\n**Analysis by {pretty_feature}**:")
            for val, fail_pct, delta in values:
                val_str = f"`{val:.2f}`" if isinstance(val, float) else f"`{val}`"
                md_parts.append(f"- **{val_str}** → found in **{fail_pct*100:.1f}%** of failures (a `{delta*100:+.1f}%` shift from the baseline).")
        return md_parts

    def _generate_cluster_markdown(self):
        if self.clustered_failures is None or self.clustered_failures.empty:
            return ["\n## No semantic failure clusters found."]
            
        md_parts = ["\n## Semantic Failure Pattern Clusters"]
        md_parts.append("Groups of failures that are semantically similar in meaning, identified using embeddings.")
        
        unique_clusters = sorted(self.clustered_failures['cluster'].unique())

        for cluster_id in unique_clusters:
            cluster_df = self.clustered_failures[self.clustered_failures['cluster'] == cluster_id]
            texts_in_cluster = cluster_df['text'].tolist()

            md_parts.append("\n---")
            if cluster_id == -1:
                md_parts.append("### Unique Failures (Noise)")
                md_parts.append(f"Found {len(texts_in_cluster)} unique failure(s) that don't fit a larger pattern.")
            else:
                try:
                    vectorizer = TfidfVectorizer(stop_words='english', max_features=5, ngram_range=(1, 2))
                    vectorizer.fit(texts_in_cluster)
                    keywords = vectorizer.get_feature_names_out()
                except ValueError:
                    keywords = ["not enough data"]

                md_parts.append(f"### Pattern Cluster {cluster_id}")
                md_parts.append(f"- **Failures in this group:** {len(texts_in_cluster)}")
                md_parts.append(f"- **Key Concepts:** `{', '.join(keywords)}`")

            md_parts.append("- **Example Failures:**")
            for text in texts_in_cluster[:3]: # Show up to 3 examples
                md_parts.append(f"  - `{text}`")
        
        return md_parts

    def generate_markdown(self):
        """Generates the full markdown report for NLP failures."""
        failure_rate = (self.failures / self.total) * 100 if self.total > 0 else 0
        md_parts = [
            f"# failprint NLP Report",
            f"- **Timestamp**: `{self.timestamp}`",
            f"- **Total Samples**: `{self.total}`",
            f"- **Failures**: `{self.failures}` ({failure_rate:.2f}%)"
        ]
        
        md_parts.extend(self._generate_segment_markdown())
        md_parts.extend(self._generate_cluster_markdown())
        
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
    def __init__(self, segments, drift_map, clustered_segments, shap_summary,
                 output, log_path, total, failures, timestamp):
        self.segments = segments
        self.drift_map = drift_map
        self.clusters = clustered_segments
        self.shap_summary = shap_summary
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

        if self.shap_summary is not None:
            md.append("\n## Top Features Driving Failures (SHAP Analysis)")
            md.append("Features are ranked by their mean absolute SHAP value across all failures.")
            md.append(self.shap_summary.to_markdown())

        return "\n".join(md)

    def write(self):
        markdown = self.generate_markdown()
        with open("reports/failprint_report.md", "w", encoding="utf-8") as f:
            f.write(markdown + "\n\n")
        with open(self.log_path, "a", encoding="utf-8") as log:
            log.write(f"[{self.timestamp}] Failures: {self.failures}/{self.total}\n")
        return markdown

def create_image_collage(image_paths: list, output_path: str, thumb_size: int = 150):
    """Creates a collage from a list of images and saves it."""
    if not image_paths:
        return

    images = [Image.open(p).convert("RGB") for p in image_paths]
    for img in images:
        img.thumbnail((thumb_size, thumb_size))

    # Determine grid size (e.g., 5 columns)
    cols = 5
    rows = math.ceil(len(images) / cols)
    width = cols * thumb_size
    height = rows * thumb_size
    
    collage = Image.new('RGB', (width, height), color='white')
    for i, img in enumerate(images):
        x = (i % cols) * thumb_size
        y = (i // cols) * thumb_size
        collage.paste(img, (x, y))
        
    collage.save(output_path)



class CvReportWriter:
    """Writes reports specifically for CV failure analysis."""
    def __init__(self, clustered_failures, cv_segments, output, log_path, total, failures, timestamp):
        self.clustered_failures = clustered_failures
        self.cv_segments = cv_segments
        self.output = output
        self.log_path = log_path
        self.total = total
        self.failures = failures
        self.timestamp = timestamp

        self.report_dir = "reports"
        os.makedirs(self.report_dir, exist_ok=True)
  
    def _generate_segment_markdown(self):
        """Generates markdown for CV feature segments."""
        if not self.cv_segments:
            return []
            
        md_parts = ["\n## Failure Segments by Image Characteristics"]
        md_parts.append("Segments where a visual property is over-represented in failures compared to the overall dataset.")
        
        for feature, values in self.cv_segments.items():
            pretty_feature = feature.replace('_', ' ').title()
            md_parts.append(f"\n**Analysis by {pretty_feature}**:")
            
            for val, fail_pct, delta in values:
                val_str = f"`{val:.2f}`" if isinstance(val, float) else f"`{val}`"
                md_parts.append(f"- **{val_str}** → found in **{fail_pct*100:.1f}%** of failures (a `{delta*100:+.1f}%` shift from the baseline).")
                
        return md_parts

    def _generate_cluster_markdown(self):
        """Generates markdown for visual failure clusters."""
        # ... (your existing implementation is perfect) ...
        if self.clustered_failures is None or self.clustered_failures.empty:
            return ["\n## No visual failure clusters found."]
        
        md_parts = ["\n## Visual Failure Pattern Clusters"]
        
        for cluster_id in sorted(self.clustered_failures['cluster'].unique()):
            cluster_df = self.clustered_failures[self.clustered_failures['cluster'] == cluster_id]
            image_paths = cluster_df['image_path'].tolist()
            
            md_parts.append("\n---")
            if cluster_id == -1:
                md_parts.append(f"### Unique Failures (Noise Points)")
                md_parts.append(f"Found {len(image_paths)} unique failures that don't fit a larger visual pattern.")
            else:
                md_parts.append(f"### Visual Pattern Cluster {cluster_id}")
                md_parts.append(f"**Failures in this group:** {len(image_paths)}")

            collage_filename = f"cluster_{cluster_id}.png"
            collage_path = os.path.join(self.report_dir, collage_filename)
            create_image_collage(image_paths[:25], collage_path)
            md_parts.append(f"\n![Cluster {cluster_id}]({collage_filename})")

        return md_parts
        
    def generate_markdown(self):
        """Generates the full markdown report for CV failures."""
        failure_rate = (self.failures / self.total) * 100 if self.total > 0 else 0
        md_parts = [
            f"# failprint CV Report",
            f"- **Timestamp**: `{self.timestamp}`",
            f"- **Total Samples**: `{self.total}`",
            f"- **Failures**: `{self.failures}` ({failure_rate:.2f}%)"
        ]

      
        md_parts.extend(self._generate_segment_markdown())
        md_parts.extend(self._generate_cluster_markdown())
        
        return "\n".join(md_parts)

    def write(self):
        """Writes the report to a file and returns the markdown string."""
        markdown = self.generate_markdown()
        report_path = os.path.join(self.report_dir, "failprint_cv_report.md")
        
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(markdown)
            
        return markdown

