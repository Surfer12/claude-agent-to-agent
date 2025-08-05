"""
UPOF Red Team Evaluation Dashboard
Real-time monitoring and visualization of evaluation results.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import yaml
from datetime import datetime
import numpy as np
from typing import Dict, List, Any
import os

class EvaluationDashboard:
    """Dashboard for monitoring red team evaluation results."""
    
    def __init__(self, config_path: str = "../config/evaluation_config.yaml"):
        """Initialize the dashboard."""
        self.config = self._load_config(config_path)
        self.setup_page()
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            st.error(f"Configuration file not found: {config_path}")
            return {}
    
    def setup_page(self):
        """Setup the Streamlit page configuration."""
        st.set_page_config(
            page_title="UPOF Red Team Evaluation Dashboard",
            page_icon="🛡️",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        st.title("🛡️ UPOF Red Team Evaluation Dashboard")
        st.markdown("Real-time monitoring of AI model vulnerability assessments")
    
    def load_evaluation_data(self, data_path: str) -> Dict[str, Any]:
        """Load evaluation data from JSON file."""
        try:
            with open(data_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            st.warning(f"Evaluation data not found: {data_path}")
            return {}
    
    def display_overview_metrics(self, data: Dict[str, Any]):
        """Display overview metrics."""
        st.header("📊 Overview Metrics")
        
        if not data:
            st.info("No evaluation data available. Run an evaluation to see metrics.")
            return
        
        # Extract metrics
        summary = data.get('overall_summary', {})
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Total Models",
                value=summary.get('total_models', 0)
            )
        
        with col2:
            st.metric(
                label="Total Runs",
                value=summary.get('total_runs', 0)
            )
        
        with col3:
            detection_rate = summary.get('vulnerability_detection_rate', 0)
            st.metric(
                label="Vulnerability Detection Rate",
                value=f"{detection_rate:.2%}"
            )
        
        with col4:
            severity_score = summary.get('average_severity_score', 0)
            st.metric(
                label="Average Severity Score",
                value=f"{severity_score:.2f}/5.0"
            )
    
    def display_model_comparison(self, data: Dict[str, Any]):
        """Display model comparison chart."""
        st.header("📈 Model Comparison")
        
        if not data or 'detailed_results' not in data:
            st.info("No detailed results available for model comparison.")
            return
        
        # Extract model data
        models_data = []
        for model_id, model_results in data['detailed_results'].items():
            models_data.append({
                'Model': model_id,
                'Detection Rate': model_results.get('vulnerability_detection_rate', 0),
                'Severity Score': model_results.get('average_severity_score', 0),
                'Total Prompts': model_results.get('total_prompts', 0)
            })
        
        if models_data:
            df = pd.DataFrame(models_data)
            
            # Create comparison chart
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=df['Model'],
                y=df['Detection Rate'],
                name='Detection Rate',
                marker_color='red'
            ))
            
            fig.add_trace(go.Bar(
                x=df['Model'],
                y=df['Severity Score'] / 5,  # Normalize to 0-1
                name='Severity Score (normalized)',
                marker_color='orange'
            ))
            
            fig.update_layout(
                title="Model Vulnerability Comparison",
                xaxis_title="Model",
                yaxis_title="Score",
                barmode='group'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No model comparison data available.")
    
    def display_test_case_analysis(self, data: Dict[str, Any]):
        """Display test case analysis."""
        st.header("🧪 Test Case Analysis")
        
        if not data or 'detailed_results' not in data:
            st.info("No detailed results available for test case analysis.")
            return
        
        # Extract test case data
        test_cases = {}
        for model_id, model_results in data['detailed_results'].items():
            for result in model_results.get('detailed_results', []):
                test_case = result.get('test_case', 'unknown')
                if test_case not in test_cases:
                    test_cases[test_case] = {
                        'total_runs': 0,
                        'vulnerabilities': 0,
                        'avg_severity': 0,
                        'severity_scores': []
                    }
                
                test_cases[test_case]['total_runs'] += 1
                if result.get('vulnerability_score', {}).get('vulnerability_detected', False):
                    test_cases[test_case]['vulnerabilities'] += 1
                
                severity = result.get('vulnerability_score', {}).get('severity', 0)
                test_cases[test_case]['severity_scores'].append(severity)
        
        # Calculate averages
        for test_case in test_cases.values():
            if test_case['severity_scores']:
                test_case['avg_severity'] = np.mean(test_case['severity_scores'])
        
        if test_cases:
            # Create test case chart
            test_case_data = []
            for test_case, data in test_cases.items():
                test_case_data.append({
                    'Test Case': test_case,
                    'Failure Rate': data['vulnerabilities'] / data['total_runs'] if data['total_runs'] > 0 else 0,
                    'Average Severity': data['avg_severity'],
                    'Total Runs': data['total_runs']
                })
            
            df = pd.DataFrame(test_case_data)
            
            fig = px.bar(
                df,
                x='Test Case',
                y='Failure Rate',
                title="Test Case Failure Rates",
                color='Average Severity',
                color_continuous_scale='Reds'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No test case data available.")
    
    def display_statistical_analysis(self, data: Dict[str, Any]):
        """Display statistical analysis."""
        st.header("📊 Statistical Analysis")
        
        if not data or 'statistical_analysis' not in data:
            st.info("No statistical analysis data available.")
            return
        
        stats_data = data['statistical_analysis']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Power Analysis")
            if 'power_analysis' in stats_data:
                power_data = stats_data['power_analysis']
                st.json(power_data)
            else:
                st.info("No power analysis data available.")
        
        with col2:
            st.subheader("Confidence Intervals")
            if 'confidence_intervals' in stats_data:
                ci_data = stats_data['confidence_intervals']
                st.json(ci_data)
            else:
                st.info("No confidence interval data available.")
    
    def display_sequential_testing(self, data: Dict[str, Any]):
        """Display sequential testing results."""
        st.header("🔄 Sequential Testing Results")
        
        if not data or 'detailed_results' not in data:
            st.info("No sequential testing data available.")
            return
        
        # Extract sequential testing data
        seq_data = []
        for model_id, model_results in data['detailed_results'].items():
            seq_results = model_results.get('sequential_testing_results', {})
            if seq_results:
                seq_data.append({
                    'Model': model_id,
                    'P-Value': seq_results.get('p_value', 1.0),
                    'Failure Rate': seq_results.get('failure_rate', 0),
                    'Total Samples': seq_results.get('total_samples', 0),
                    'Continue Testing': seq_results.get('continue_testing', True)
                })
        
        if seq_data:
            df = pd.DataFrame(seq_data)
            
            # Create sequential testing visualization
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=df['Model'],
                y=df['P-Value'],
                mode='markers',
                name='P-Value',
                marker=dict(
                    size=10,
                    color=df['P-Value'],
                    colorscale='RdYlGn_r',
                    showscale=True
                )
            ))
            
            fig.update_layout(
                title="Sequential Testing P-Values",
                xaxis_title="Model",
                yaxis_title="P-Value",
                yaxis=dict(range=[0, 1])
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display table
            st.subheader("Sequential Testing Details")
            st.dataframe(df)
        else:
            st.info("No sequential testing results available.")
    
    def display_recommendations(self, data: Dict[str, Any]):
        """Display recommendations."""
        st.header("💡 Recommendations")
        
        if not data or 'recommendations' not in data:
            st.info("No recommendations available.")
            return
        
        recommendations = data['recommendations']
        
        for i, rec in enumerate(recommendations, 1):
            st.markdown(f"**{i}.** {rec}")
    
    def display_logs(self, data: Dict[str, Any]):
        """Display evaluation logs."""
        st.header("📝 Evaluation Logs")
        
        if not data or 'logs' not in data:
            st.info("No evaluation logs available.")
            return
        
        logs = data['logs']
        
        # Create logs dataframe
        logs_data = []
        for log in logs:
            logs_data.append({
                'Timestamp': log.get('timestamp', ''),
                'Model': log.get('model_id', ''),
                'Prompt ID': log.get('prompt_id', ''),
                'Vulnerability Detected': log.get('vulnerability_score', {}).get('vulnerability_detected', False),
                'Severity': log.get('vulnerability_score', {}).get('severity', 0)
            })
        
        if logs_data:
            df = pd.DataFrame(logs_data)
            
            # Filter options
            st.subheader("Log Filters")
            col1, col2 = st.columns(2)
            
            with col1:
                show_vulnerabilities = st.checkbox("Show only vulnerabilities", value=True)
            
            with col2:
                min_severity = st.slider("Minimum severity", 0, 5, 0)
            
            # Apply filters
            if show_vulnerabilities:
                df = df[df['Vulnerability Detected'] == True]
            
            df = df[df['Severity'] >= min_severity]
            
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No log data available.")
    
    def run_dashboard(self, data_path: str = None):
        """Run the dashboard."""
        # Sidebar for navigation
        st.sidebar.title("Navigation")
        
        if data_path:
            data = self.load_evaluation_data(data_path)
        else:
            # Allow file upload
            uploaded_file = st.sidebar.file_uploader(
                "Upload evaluation data (JSON)",
                type=['json']
            )
            
            if uploaded_file:
                data = json.load(uploaded_file)
            else:
                data = {}
        
        # Navigation
        page = st.sidebar.selectbox(
            "Select Page",
            ["Overview", "Model Comparison", "Test Case Analysis", 
             "Statistical Analysis", "Sequential Testing", "Recommendations", "Logs"]
        )
        
        # Display selected page
        if page == "Overview":
            self.display_overview_metrics(data)
        elif page == "Model Comparison":
            self.display_model_comparison(data)
        elif page == "Test Case Analysis":
            self.display_test_case_analysis(data)
        elif page == "Statistical Analysis":
            self.display_statistical_analysis(data)
        elif page == "Sequential Testing":
            self.display_sequential_testing(data)
        elif page == "Recommendations":
            self.display_recommendations(data)
        elif page == "Logs":
            self.display_logs(data)
        
        # Footer
        st.sidebar.markdown("---")
        st.sidebar.markdown("**UPOF Red Team Evaluation Framework v2.0**")
        st.sidebar.markdown(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

def main():
    """Main function to run the dashboard."""
    dashboard = EvaluationDashboard()
    dashboard.run_dashboard()

if __name__ == "__main__":
    main()