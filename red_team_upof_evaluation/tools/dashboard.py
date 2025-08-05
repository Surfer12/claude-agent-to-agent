"""
UPOF Red Team Evaluation Dashboard

Streamlit dashboard for real-time analysis and visualization of evaluation results.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import yaml
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Page configuration
st.set_page_config(
    page_title="UPOF Red Team Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

class UPOFDashboard:
    """Main dashboard class for UPOF red team evaluation results."""
    
    def __init__(self):
        self.reports_dir = Path("reports")
        self.logs_dir = Path("logs")
        self.config_path = Path("configs/test_config.yaml")
        
    def load_config(self):
        """Load configuration file."""
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        return {}
    
    def load_reports(self):
        """Load available evaluation reports."""
        if not self.reports_dir.exists():
            return []
        
        reports = []
        for report_file in self.reports_dir.glob("*.json"):
            try:
                with open(report_file, 'r') as f:
                    report = json.load(f)
                    report['filename'] = report_file.name
                    reports.append(report)
            except Exception as e:
                st.error(f"Error loading {report_file}: {e}")
        
        return sorted(reports, key=lambda x: x.get('timestamp', ''), reverse=True)
    
    def load_logs(self, hours_back=24):
        """Load recent logs."""
        if not self.logs_dir.exists():
            return []
        
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        logs = []
        
        for log_file in self.logs_dir.glob("*.jsonl"):
            try:
                with open(log_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            log_entry = json.loads(line)
                            log_time = datetime.fromisoformat(log_entry['timestamp'].replace('Z', '+00:00'))
                            if log_time.replace(tzinfo=None) > cutoff_time:
                                logs.append(log_entry)
            except Exception as e:
                st.error(f"Error loading {log_file}: {e}")
        
        return sorted(logs, key=lambda x: x['timestamp'], reverse=True)
    
    def create_overview_metrics(self, report):
        """Create overview metrics display."""
        if not report or 'results' not in report:
            st.warning("No results data available")
            return
        
        results = report['results']
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Tests",
                results.get('total_tests', 0),
                help="Total number of tests executed"
            )
        
        with col2:
            vulnerability_rate = 0
            if 'statistical_summary' in results and 'overall' in results['statistical_summary']:
                vulnerability_rate = results['statistical_summary']['overall'].get('failure_rate', 0) * 100
            
            st.metric(
                "Vulnerability Rate",
                f"{vulnerability_rate:.1f}%",
                help="Percentage of tests that detected vulnerabilities"
            )
        
        with col3:
            models_tested = len(results.get('by_model', {}))
            st.metric(
                "Models Tested",
                models_tested,
                help="Number of AI models evaluated"
            )
        
        with col4:
            statistical_power = 0
            if 'statistical_summary' in results and 'overall' in results['statistical_summary']:
                statistical_power = results['statistical_summary']['overall'].get('statistical_power', 0) * 100
            
            st.metric(
                "Statistical Power",
                f"{statistical_power:.1f}%",
                help="Statistical power of the evaluation"
            )
    
    def create_vulnerability_analysis(self, report):
        """Create vulnerability analysis visualizations."""
        if not report or 'results' not in report:
            return
        
        results = report['results']
        vuln_analysis = results.get('vulnerability_analysis', {})
        
        if not vuln_analysis or vuln_analysis.get('total_vulnerabilities', 0) == 0:
            st.info("No vulnerabilities detected in this evaluation")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Severity Distribution")
            
            severity_dist = vuln_analysis.get('severity_distribution', {})
            if severity_dist:
                severity_labels = ["None", "Low", "Low-Med", "Medium", "High", "Critical"]
                
                df_severity = pd.DataFrame([
                    {"Severity": severity_labels[int(k)], "Count": v, "Level": int(k)}
                    for k, v in severity_dist.items()
                ])
                
                fig = px.bar(
                    df_severity,
                    x="Severity",
                    y="Count",
                    color="Level",
                    color_continuous_scale="Reds",
                    title="Vulnerability Severity Distribution"
                )
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Vulnerabilities by Category")
            
            by_category = vuln_analysis.get('by_category', {})
            if by_category:
                df_category = pd.DataFrame([
                    {
                        "Category": cat.replace('_', ' ').title(),
                        "Count": data['count'],
                        "Avg Severity": data['avg_severity']
                    }
                    for cat, data in by_category.items()
                ])
                
                fig = px.scatter(
                    df_category,
                    x="Count",
                    y="Avg Severity",
                    size="Count",
                    hover_name="Category",
                    title="Category Analysis: Count vs Average Severity"
                )
                st.plotly_chart(fig, use_container_width=True)
    
    def create_model_comparison(self, report):
        """Create model comparison visualizations."""
        if not report or 'results' not in report:
            return
        
        by_model = report['results'].get('by_model', {})
        if len(by_model) < 2:
            st.info("Need at least 2 models for comparison")
            return
        
        st.subheader("Model Performance Comparison")
        
        # Prepare data for comparison
        model_data = []
        for model_name, model_results in by_model.items():
            model_data.append({
                "Model": model_name,
                "Vulnerability Rate": model_results.get('vulnerability_rate', 0) * 100,
                "Average Severity": model_results.get('average_severity', 0),
                "Total Tests": model_results.get('total_tests', 0),
                "Response Time": model_results.get('average_execution_time', 0)
            })
        
        df_models = pd.DataFrame(model_data)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(
                df_models,
                x="Model",
                y="Vulnerability Rate",
                title="Vulnerability Rate by Model (%)",
                color="Vulnerability Rate",
                color_continuous_scale="Reds"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.scatter(
                df_models,
                x="Response Time",
                y="Average Severity",
                size="Total Tests",
                hover_name="Model",
                title="Response Time vs Average Severity"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Model comparison table
        st.subheader("Detailed Model Comparison")
        st.dataframe(df_models, use_container_width=True)
    
    def create_prompt_analysis(self, report):
        """Create prompt template analysis."""
        if not report or 'results' not in report:
            return
        
        vuln_analysis = report['results'].get('vulnerability_analysis', {})
        most_vulnerable = vuln_analysis.get('most_vulnerable_prompts', [])
        
        if not most_vulnerable:
            st.info("No vulnerable prompts identified")
            return
        
        st.subheader("Most Vulnerable Prompt Templates")
        
        df_prompts = pd.DataFrame(most_vulnerable)
        
        # Vulnerability rate chart
        fig = px.bar(
            df_prompts.head(10),
            x="vulnerability_rate",
            y="prompt_id",
            orientation='h',
            title="Top 10 Most Vulnerable Prompts",
            color="average_severity",
            color_continuous_scale="Reds"
        )
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed table
        st.subheader("Prompt Vulnerability Details")
        formatted_df = df_prompts.copy()
        formatted_df['vulnerability_rate'] = (formatted_df['vulnerability_rate'] * 100).round(1)
        formatted_df['average_severity'] = formatted_df['average_severity'].round(2)
        
        st.dataframe(
            formatted_df.rename(columns={
                'prompt_id': 'Prompt ID',
                'vulnerability_rate': 'Vulnerability Rate (%)',
                'average_severity': 'Average Severity',
                'total_tests': 'Total Tests',
                'vulnerable_tests': 'Vulnerable Tests'
            }),
            use_container_width=True
        )
    
    def create_statistical_analysis(self, report):
        """Create statistical analysis section."""
        if not report or 'results' not in report:
            return
        
        stat_summary = report['results'].get('statistical_summary', {})
        overall_stats = stat_summary.get('overall', {})
        
        if not overall_stats:
            st.warning("No statistical summary available")
            return
        
        st.subheader("Statistical Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Failure Rate",
                f"{overall_stats.get('failure_rate', 0) * 100:.2f}%",
                help="Observed failure rate across all tests"
            )
        
        with col2:
            ci = overall_stats.get('confidence_interval', [0, 0])
            st.metric(
                "95% Confidence Interval",
                f"[{ci[0]*100:.1f}%, {ci[1]*100:.1f}%]",
                help="95% confidence interval for failure rate"
            )
        
        with col3:
            p_value = overall_stats.get('p_value', 1.0)
            significance = "Significant" if p_value < 0.05 else "Not Significant"
            st.metric(
                "Statistical Significance",
                significance,
                f"p-value: {p_value:.4f}" if p_value else "N/A"
            )
        
        # Power analysis
        st.subheader("Power Analysis")
        
        power_data = {
            "Metric": ["Statistical Power", "Effect Size", "Total Runs"],
            "Value": [
                f"{overall_stats.get('statistical_power', 0) * 100:.1f}%",
                f"{overall_stats.get('effect_size', 0):.3f}",
                overall_stats.get('total_runs', 0)
            ]
        }
        
        st.table(pd.DataFrame(power_data))
    
    def create_real_time_monitoring(self, logs):
        """Create real-time monitoring section."""
        if not logs:
            st.info("No recent activity")
            return
        
        st.subheader("Recent Activity (Last 24 Hours)")
        
        # Convert logs to DataFrame
        df_logs = pd.DataFrame(logs)
        df_logs['timestamp'] = pd.to_datetime(df_logs['timestamp'])
        
        # Activity over time
        df_logs['hour'] = df_logs['timestamp'].dt.floor('H')
        activity_by_hour = df_logs.groupby('hour').size().reset_index(name='count')
        
        fig = px.line(
            activity_by_hour,
            x='hour',
            y='count',
            title="Test Activity Over Time",
            markers=True
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Recent vulnerabilities
        recent_vulns = df_logs[df_logs['flags'].apply(lambda x: x.get('undesired', False))]
        
        if not recent_vulns.empty:
            st.subheader("Recent Vulnerabilities Detected")
            
            vuln_summary = recent_vulns.groupby(['model_id', 'prompt_id']).size().reset_index(name='count')
            
            fig = px.treemap(
                vuln_summary,
                path=['model_id', 'prompt_id'],
                values='count',
                title="Vulnerability Distribution by Model and Prompt"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Recent test results table
        st.subheader("Latest Test Results")
        
        display_logs = df_logs.head(20)[['timestamp', 'model_id', 'prompt_id', 'severity_score', 'flags']]
        display_logs['vulnerable'] = display_logs['flags'].apply(lambda x: x.get('undesired', False))
        display_logs = display_logs.drop('flags', axis=1)
        
        st.dataframe(display_logs, use_container_width=True)
    
    def create_configuration_panel(self, config):
        """Create configuration panel."""
        st.subheader("Current Configuration")
        
        if not config:
            st.warning("No configuration loaded")
            return
        
        # Statistical configuration
        with st.expander("Statistical Configuration"):
            stats_config = config.get('statistics', {})
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Target Power:**", stats_config.get('target_power', 0.8))
                st.write("**Base Failure Rate:**", stats_config.get('base_failure_rate', 0.05))
                st.write("**Confidence Level:**", stats_config.get('confidence_level', 0.95))
            
            with col2:
                st.write("**Min Runs per Prompt:**", stats_config.get('min_runs_per_prompt', 50))
                st.write("**Max Runs per Prompt:**", stats_config.get('max_runs_per_prompt', 200))
                st.write("**Vulnerability Threshold:**", stats_config.get('vulnerability_threshold', 0.10))
        
        # Model configuration
        with st.expander("Model Configuration"):
            models = config.get('models', [])
            if models:
                df_models = pd.DataFrame(models)
                st.dataframe(df_models, use_container_width=True)
            else:
                st.info("No models configured")
        
        # Test sources
        with st.expander("Test Sources"):
            sources = config.get('test_sources', [])
            if sources:
                for source in sources:
                    st.write(f"**{source['source']}:** {source['description']}")
                    st.write(f"Theorems: {len(source.get('theorems', []))}")
            else:
                st.info("No test sources configured")

def main():
    """Main dashboard function."""
    st.title("🔍 UPOF Red Team Evaluation Dashboard")
    st.markdown("Real-time monitoring and analysis of AI model vulnerability assessments")
    
    dashboard = UPOFDashboard()
    
    # Sidebar
    st.sidebar.title("Navigation")
    
    # Load data
    reports = dashboard.load_reports()
    config = dashboard.load_config()
    logs = dashboard.load_logs()
    
    # Report selection
    if reports:
        report_options = [f"{r['filename']} ({r.get('timestamp', 'Unknown')})" for r in reports]
        selected_report_idx = st.sidebar.selectbox(
            "Select Evaluation Report",
            range(len(report_options)),
            format_func=lambda x: report_options[x]
        )
        selected_report = reports[selected_report_idx]
    else:
        st.sidebar.warning("No evaluation reports found")
        selected_report = None
    
    # Page selection
    pages = [
        "📊 Overview",
        "🎯 Vulnerability Analysis", 
        "🔄 Model Comparison",
        "📝 Prompt Analysis",
        "📈 Statistical Analysis",
        "⏱️ Real-time Monitoring",
        "⚙️ Configuration"
    ]
    
    selected_page = st.sidebar.radio("Select View", pages)
    
    # Refresh button
    if st.sidebar.button("🔄 Refresh Data"):
        st.rerun()
    
    # Main content
    if selected_page == "📊 Overview":
        st.header("Evaluation Overview")
        if selected_report:
            dashboard.create_overview_metrics(selected_report)
            
            # Quick summary
            st.subheader("Executive Summary")
            if 'results' in selected_report:
                results = selected_report['results']
                total_tests = results.get('total_tests', 0)
                vulnerability_analysis = results.get('vulnerability_analysis', {})
                total_vulns = vulnerability_analysis.get('total_vulnerabilities', 0)
                
                if total_vulns > 0:
                    vuln_rate = (total_vulns / total_tests * 100) if total_tests > 0 else 0
                    st.warning(f"⚠️ {total_vulns} vulnerabilities detected across {total_tests} tests ({vuln_rate:.1f}% vulnerability rate)")
                else:
                    st.success("✅ No vulnerabilities detected in this evaluation")
    
    elif selected_page == "🎯 Vulnerability Analysis":
        st.header("Vulnerability Analysis")
        if selected_report:
            dashboard.create_vulnerability_analysis(selected_report)
    
    elif selected_page == "🔄 Model Comparison":
        st.header("Model Comparison")
        if selected_report:
            dashboard.create_model_comparison(selected_report)
    
    elif selected_page == "📝 Prompt Analysis":
        st.header("Prompt Template Analysis")
        if selected_report:
            dashboard.create_prompt_analysis(selected_report)
    
    elif selected_page == "📈 Statistical Analysis":
        st.header("Statistical Analysis")
        if selected_report:
            dashboard.create_statistical_analysis(selected_report)
    
    elif selected_page == "⏱️ Real-time Monitoring":
        st.header("Real-time Monitoring")
        dashboard.create_real_time_monitoring(logs)
    
    elif selected_page == "⚙️ Configuration":
        st.header("Configuration")
        dashboard.create_configuration_panel(config)
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("**UPOF Red Team Dashboard**")
    st.sidebar.markdown("AI Safety Evaluation Framework")

if __name__ == "__main__":
    main()