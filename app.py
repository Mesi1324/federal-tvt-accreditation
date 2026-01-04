import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime

# ==========================================
# 1. AI-DRIVEN NARRATIVE ENGINE
# ==========================================
class AINarrativeGenerator:
    """Generates ABET Self-Study Report (SSR) text based on MCDM data."""
    @staticmethod
    def generate_report(program_name, score, status, strengths, weaknesses):
        # In production, this would call OpenAI/Anthropic API
        narrative = f"""
        ### ABET Self-Study Executive Summary: {program_name}
        **Overall Readiness Index:** {score:.2f}
        **Accreditation Status:** {status}
        
        **Analysis of Student Outcomes:**
        The {program_name} program demonstrates a {strengths[0]} performance in {strengths[1]}. 
        However, the current assessment indicates a critical gap in {weaknesses[1]}, where the 
        department scored {weaknesses[0]}. 
        
        **Improvement Action Plan:**
        Based on the MCDM Gap Analysis, the institute will reallocate 15% of the next 
        fiscal budget to address the deficiencies in {weaknesses[1]} to ensure full 
        compliance with international ABET Student Outcome standards.
        """
        return narrative

# ==========================================
# 2. NATIONAL BENCHMARKING DATA
# ==========================================
def fetch_national_data():
    return pd.DataFrame({
        "Institute": ["Federal TVT Central", "TVT North Region", "TVT South Region", "TVT Eastern Tech", "TVT Western Vocational"],
        "Avg_Readiness": [82.5, 74.2, 88.9, 69.5, 78.1],
        "Safety_Compliance": [92, 80, 95, 70, 85],
        "Employment_Rate": [88, 75, 92, 65, 80]
    })

# ==========================================
# 3. CORE ENGINE (UPGRADED)
# ==========================================
class AdvancedMCDMEngine:
    def __init__(self, data, weights, types):
        self.matrix = data
        self.weights = weights
        self.types = types

    def calculate_cradis(self, alt_weights=None):
        w = alt_weights if alt_weights is not None else self.weights
        norm = self.matrix / self.matrix.max(axis=0) # Simplified linear norm
        weighted = norm * w
        s_ideal, s_anti = weighted.max(axis=0), weighted.min(axis=0)
        d_ideal = np.sum(np.abs(s_ideal - weighted), axis=1)
        d_anti = np.sum(np.abs(weighted - s_anti), axis=1)
        return d_anti / (d_ideal + d_anti)

# ==========================================
# 4. STREAMLIT GUI
# ==========================================
def main():
    st.set_page_config(page_title="Federal TVT Intelligence", layout="wide")
    st.title("🛡️ Federal TVT Institutional Intelligence Platform")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Program Ranking", "Sensitivity Analysis", "AI Narrative", "National Dashboard"])

    # Shared Data Setup
    programs = ["Automotive", "ICT", "Civil", "Electrical", "Renewable"]
    criteria = ["Student_Outcomes", "Faculty_Cert", "Labs", "Industry", "Safety", "Cost"]
    raw_data = np.array([
        [85, 90, 95, 12, 98, 45], [92, 85, 70, 15, 85, 38],
        [78, 80, 88, 10, 40, 52], [88, 82, 95, 8, 88, 41],
        [70, 75, 80, 90, 75, 45]
    ])
    weights = np.array([0.25, 0.15, 0.15, 0.10, 0.25, 0.10])
    types = ['max']*5 + ['min']

    engine = AdvancedMCDMEngine(raw_data, weights, types)
    scores = engine.calculate_cradis()
    df = pd.DataFrame(raw_data, columns=criteria, index=programs)
    df['Readiness_Score'] = scores

 # --- TAB 1: RANKING ---
    with tab1:
        st.subheader("Accreditation Readiness Scores")
        # Display the data table
        st.dataframe(df.style.highlight_max(axis=0, subset=['Readiness_Score']))
        
        # FIXED: Changed px.radar to px.line_polar
        fig = px.line_polar(
            df.reset_index(), 
            r='Readiness_Score', 
            theta='index', 
            line_close=True,
            title="Comparative Program Readiness"
        )
        st.plotly_chart(fig, use_container_width=True)

    # --- TAB 2: SENSITIVITY HEATMAP ---
    with tab2:
        st.subheader("Weight Sensitivity Heatmap")
        st.write("How does changing the importance of a criterion change the #1 Rank?")
        
        sensitivity_matrix = []
        variations = np.linspace(0.1, 0.5, 5) 
        
        for v in variations:
            row = []
            for j in range(len(criteria)):
                temp_weights = weights.copy()
                temp_weights[j] = v
                temp_weights /= temp_weights.sum()
                new_scores = engine.calculate_cradis(temp_weights)
                winner_idx = np.argmax(new_scores)
                row.append(winner_idx)
            sensitivity_matrix.append(row)
        
        # FIXED: Robust Matplotlib figure handling for Streamlit Cloud
        fig_heat, ax = plt.subplots(figsize=(10, 5))
        sns.heatmap(sensitivity_matrix, annot=True, xticklabels=criteria, yticklabels=[f"{round(v,2)}" for v in variations], ax=ax, cmap="YlGnBu")
        plt.xlabel("Criterion adjusted")
        plt.ylabel("Weight Variance")
        st.pyplot(fig_heat)
        plt.close(fig_heat) # Clean up memory
        
        st.info("Legend: Numbers inside boxes represent the index of the winning program. 0=Automotive, 1=ICT, etc. If the number changes in a column, the ranking is sensitive.")

    # --- TAB 3: AI NARRATIVE ---
    with tab3:
        st.subheader("AI-Generated ABET Self-Study Draft")
        target_prog = st.selectbox("Select Program to Generate Report", programs)
        prog_score = df.loc[target_prog, 'Readiness_Score']
        
        # Get actual data row for context
        row_vals = df.loc[target_prog, criteria]
        strength_col = row_vals.idxmax()
        weakness_col = row_vals.idxmin()
        
        report_text = AINarrativeGenerator.generate_report(
            target_prog, 
            prog_score, 
            "Ready" if prog_score > 0.7 else "Action Required", 
            [row_vals[strength_col], strength_col], 
            [row_vals[weakness_col], weakness_col]
        )
        st.markdown(report_text)

    # --- TAB 4: NATIONAL DASHBOARD (Scientific Edge) ---
    with tab4:
        st.subheader("Federal Oversight: National TVT Benchmarking")
        nat_df = fetch_national_data()
        
        col1, col2 = st.columns(2)
        with col1:
            fig_nat = px.scatter(nat_df, x="Avg_Readiness", y="Safety_Compliance", size="Employment_Rate", text="Institute", title="National TVT Quality Landscape")
            st.plotly_chart(fig_nat)
        with col2:
            st.metric("Highest National Readiness", nat_df.Institute[nat_df.Avg_Readiness.idxmax()], f"{nat_df.Avg_Readiness.max()}%")
            st.write("This dashboard allows the Federal Board to identify regional clusters of excellence and programs in need of intervention.")

if __name__ == "__main__":

    main()
