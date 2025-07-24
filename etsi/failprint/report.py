import os


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

    def generate_html(self):
        import plotly.graph_objs as go
        import pandas as pd
        from jinja2 import Environment, FileSystemLoader, select_autoescape

        # Summary
        misclass_pct = round((self.failures / self.total) * 100, 2) if self.total else 0

        # Top features chart (bar plot)
        feature_names = []
        feature_impacts = []
        for feat, vals in self.segments.items():
            feature_names.append(feat)
            # Use max delta as impact
            feature_impacts.append(max([abs(delta) for _, _, delta in vals]) if vals else 0)
        chart_html = ""
        plotly_scripts = ""
        if feature_names:
            fig = go.Figure([go.Bar(x=feature_names, y=feature_impacts, marker_color='#0984e3')])
            fig.update_layout(title="Top Features Influencing Failures", xaxis_title="Feature", yaxis_title="Impact (Δ)")
            chart_html = fig.to_html(full_html=False, include_plotlyjs='cdn')
            # Extract plotly script for embedding
            plotly_scripts = ''
        
        # Cluster-wise stats (as HTML tables)
        cluster_stats = ""
        if self.clusters is not None and len(self.clusters) > 0:
            for i, cluster in enumerate(self.clusters):
                df = pd.DataFrame(cluster)
                cluster_stats += f'<div class="cluster-block"><b>Cluster {i+1}</b><br>'
                cluster_stats += df.to_html(classes='feature-table', index=False)
                cluster_stats += '</div>'

        # Failure segments (highlighted)
        failure_segments = ""
        for feat, vals in self.segments.items():
            failure_segments += f'<b>{feat}</b>:<ul>'
            for val, fail_pct, delta in vals:
                style = 'class="highlight"' if fail_pct > 0.5 else ''
                failure_segments += f'<li {style}>{val} → {fail_pct*100:.1f}% in failures (Δ +{delta*100:.1f}%)</li>'
            failure_segments += '</ul>'

        # Jinja2 template rendering
        env = Environment(
            loader=FileSystemLoader(os.path.join(os.path.dirname(__file__), 'templates')),
            autoescape=select_autoescape(['html'])
        )
        template = env.get_template('failprint_report.html')
        html = template.render(
            timestamp=self.timestamp,
            total=self.total,
            failures=self.failures,
            misclass_pct=misclass_pct,
            feature_chart=chart_html,
            plotly_scripts=plotly_scripts,
            cluster_stats=cluster_stats,
            failure_segments=failure_segments
        )
        return html

    def write(self):
        if self.output == "html":
            html = self.generate_html()
            with open("reports/failprint_report.html", "w", encoding="utf-8") as f:
                f.write(html)
            with open(self.log_path, "a", encoding="utf-8") as log:
                log.write(f"[{self.timestamp}] Failures: {self.failures}/{self.total}\n")
            return html
        else:
            markdown = self.generate_markdown()
            with open("reports/failprint_report.md", "w", encoding="utf-8") as f:
                f.write(markdown + "\n\n")
            with open(self.log_path, "a", encoding="utf-8") as log:
                log.write(f"[{self.timestamp}] Failures: {self.failures}/{self.total}\n")
            return markdown
