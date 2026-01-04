import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import google.generativeai as genai
from datetime import datetime

# ==========================================
# 1. AI NARRATIVE ENGINE
# ==========================================
class AINarrativeGenerator:
    """Generates ABET/ETA reports using Google Gemini AI."""
    @staticmethod
    def generate_report(program_name, score, status, strength, weakness):
        # Check for API Key
        api_key = st.secrets.get("GOOGLE_API_KEY", None)
        
        if not api_key:
            return (
                "### ⚠️ API Key Missing\n"
                "To generate a real AI report, please add `GOOGLE_API_KEY` to your Streamlit Secrets.\n\n"
                "**Simulated Summary:**\n"
                f"The **{program_name}** department at FDRE TVT Institute has achieved a readiness score of **{score:.2f}**. "
                f"Its primary strength is **{strength[1]}**, while urgent attention is needed in **{weakness[1]}**."
            )

        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-1.5-flash')
            
            # PROMPT TAILORED FOR FDRE TVT INSTITUTE
            prompt = (
                "You are an expert academic accreditation consultant for the FDRE Technical and Vocational Training Institute in Ethiopia. "
                "The institute adheres to International ABET protocols and National ETA standards (Ministry of Labor and Skills). "
                "Write a formal 'Self-Study Report' executive summary for the following department:\n\n"
                "Department: {name}\n"
                "Overall Readiness Score: {score:.2f} (Scale 0-1)\n"
                "Status: {status}\n"
                "Top Strength: {str_name} (Score: {str_val})\n"
                "Critical Weakness: {weak_name} (Score: {weak_val})\n\n"
                "Please structure the response with these headers:\n"
                "1. Executive Summary (Context: FDRE TVT Institute)\n"
                "2. ABET Student Outcome Compliance\n"
                "3. National COC/OCA Competency Analysis (Ethiopian Occupational Standards)\n"
                "4. Kaizen & 5S Implementation Strategy"
            ).format(
                name=program_name, 
                score=score, 
                status=status, 
                str_name=strength[1], 
                str_val=strength[0], 
                weak_name=weakness[1], 
                weak_val=weakness[0]
            )
            
            response = model.generate_content(prompt)
            return response.text
            
        except Exception as e:
            return f"⚠️ AI Service Error: {str(e)}"

# ==========================================
# 2. NATIONAL STANDARDS & TARGETS
# ==========================================
def fetch_national_targets():
    """
    Since there is only one institute, we compare Departments against 
    National Ministry (MOLS) Targets rather than other institutes.
    """
    # These represent the 'Ideal' National Targets for Ethiopian TVT
    return {
        "Min_COC_Pass_Rate": 85.0,
        "Min_Coop_Rate": 80.0,
        "Target_Kaizen_Score": 90.0
    }

# ==========================================
# 3. ADVANCED MCDM ENGINE (CRADIS)
# ==========================================
class AdvancedMCDMEngine:
    def __init__(self, data, weights, types):
        self.matrix = data
        self.weights = weights
        self.types = types

    def calculate_cradis(self, alt_weights=None):
        w = alt_weights if alt_weights is not None else self.weights
        # Normalization with safety epsilon
        norm = np.zeros_like(self.matrix, dtype=float)
        for j in range(self.matrix.shape[1]):
            if self.types[j] == 'max':
                norm[:, j] = self.matrix[:, j] / (np.max(self.matrix[:, j]) + 1e-9)
            else:
                norm[:, j] = np.min(self.matrix[:, j]) / (self.matrix[:, j] + 1e-9)
        
        weighted = norm * w
        s_ideal = np.max(weighted, axis=0)
        s_anti = np.min(weighted, axis=0)
        
        d_ideal = np.sum(np.abs(s_ideal - weighted), axis=1)
        d_anti = np.sum(np.abs(weighted - s_anti), axis=1)
        
        return d_anti / (d_ideal + d_anti + 1e-9)

# ==========================================
# 4. STREAMLIT GUI (MAIN APP)
# ==========================================
def main():
    st.set_page_config(page_title="FDRE TVT Accreditation Portal", layout="wide", page_icon="🇪🇹")
    
    # Header Section
    c1, c2 = st.columns([1, 5])
    with c1:
        # Placeholder for Institute Logo
        st.write("🇪🇹") 
    with c2:
        st.title("FDRE Technical & Vocational Training Institute")
        st.subheader("Integrated Accreditation & Quality Assurance System")
    
    st.markdown("---")

    # --- SHARED DATA SETUP ---
    # Specific Departments requested + 2 supplementary for matrix robustness
    programs = [
        "Building Construction Tech", 
        "Automotive Technology", 
        "Manufacturing Technology", 
        "Electrical/Electronics", 
        "ICT & Database Admin"
    ]
    
    # CRITERIA DEFINITIONS (Contextualized):
    # 1. ABET_SO: Student Outcomes (International Standard)
    # 2. COC_Pass_Rate: Certificate of Competency (National ETA Standard)
    # 3. Coop_Training: Industry Cooperative Training Rate
    # 4. Trainer_Cert: C-Level certified trainers
    # 5. Kaizen_Audit: 5S and Continuous Improvement Audit Score
    # 6. Cost_Efficiency: Unit cost per trainee
    
    criteria = ["ABET_SO_Compliance", "COC_Pass_Rate", "Coop_Training_Rate", "Trainer_Cert_Level", "Kaizen_Audit", "Unit_Cost"]
    
    # Synthetic Data reflecting realistic TVT scenarios
    # Building: High Kaizen, Med Cost
    # Auto: High Cost, High Coop, Med COC
    # Mfg: High COC, High Cost
    raw_data = np.array([
        [88, 92, 85, 90, 95, 4800], # Building Construction
        [85, 80, 95, 88, 85, 6200], # Automotive
        [90, 94, 88, 92, 90, 5800], # Manufacturing
        [82, 85, 80, 85, 75, 4100], # Electrical
        [94, 88, 75, 80, 82, 3500]  # ICT
    ])
    
    # Weights (Prioritizing COC and Coop for National Context)
    base_weights = np.array([0.20, 0.25, 0.20, 0.15, 0.10, 0.10])
    types = ['max', 'max', 'max', 'max', 'max', 'min']

    # Initialize Engine
    engine = AdvancedMCDMEngine(raw_data, base_weights, types)
    scores = engine.calculate_cradis()
    
    df = pd.DataFrame(raw_data, columns=criteria, index=programs)
    df['Readiness_Score'] = scores
    df['Rank'] = df['Readiness_Score'].rank(ascending=False)

    # --- TABS LAYOUT ---
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Dept. Readiness", 
        "⚖️ Bias/Sensitivity", 
        "📝 AI Self-Study", 
        "🎯 National Compliance"
    ])

    # --- TAB 1: RANKING & GAP ANALYSIS ---
    with tab1:
        st.markdown("### 🏆 Accreditation Readiness Ranking")
        col_a, col_b = st.columns([2, 1])
        
        with col_a:
            # Highlight top performers
            st.dataframe(
                df.sort_values("Readiness_Score", ascending=False).style.background_gradient(subset=['Readiness_Score'], cmap="Blues"),
                use_container_width=True
            )
        
        with col_b:
            st.markdown("#### Gap Analysis (Radar)")
            # Using px.line_polar
            fig = px.line_polar(
                df.reset_index(), 
                r='Readiness_Score', 
                theta='index', 
                line_close=True,
                title="Departmental Readiness Profile",
                range_r=[0, 1]
            )
            st.plotly_chart(fig, use_container_width=True)
            
        st.info("ℹ️ **Protocol Note:** Building Construction, Automotive, and Manufacturing are currently prioritized for the upcoming ABET site visit.")

    # --- TAB 2: SENSITIVITY HEATMAP ---
    with tab2:
        st.subheader("Weight Sensitivity Analysis")
        st.markdown("Analyze if the **Department Ranking** changes based on the importance given to specific criteria (e.g., favoring *COC Pass Rate* vs *Kaizen*).")
        
        sensitivity_matrix = []
        variations = np.linspace(0.1, 0.5, 5) 
        
        for v in variations:
            row = []
            for j in range(len(criteria)):
                temp_weights = base_weights.copy()
                temp_weights[j] = v
                temp_weights /= temp_weights.sum()
                new_scores = engine.calculate_cradis(temp_weights)
                winner_idx = np.argmax(new_scores)
                row.append(winner_idx)
            sensitivity_matrix.append(row)
        
        fig_heat, ax = plt.subplots(figsize=(10, 4))
        sns.heatmap(
            sensitivity_matrix, 
            annot=True, 
            xticklabels=criteria, 
            yticklabels=[f"{v:.1f}" for v in variations], 
            ax=ax, 
            cmap="YlGnBu",
            cbar_kws={'label': 'Winning Dept Index'}
        )
        st.pyplot(fig_heat)
        plt.close(fig_heat)
        st.caption("Legend: 0=Building, 1=Auto, 2=Mfg. If the number remains constant, the lead department is robust.")

    # --- TAB 3: AI NARRATIVE ---
    with tab3:
        st.subheader("🤖 AI-Generated Self-Study (ABET & ETA)")
        
        c1, c2 = st.columns([1, 2])
        with c1:
            target_prog = st.selectbox("Select Department", programs)
            prog_score = df.loc[target_prog, 'Readiness_Score']
            
            row_vals = df.loc[target_prog, criteria]
            # Drop cost for strength calculation
            numeric_criteria = row_vals.drop('Unit_Cost') 
            
            str_col = numeric_criteria.idxmax()
            weak_col = numeric_criteria.idxmin()
            
            st.metric("Readiness Score", f"{prog_score:.3f}")
            st.write(f"**Strength:** {str_col}")
            st.write(f"**Weakness:** {weak_col}")

        with c2:
            if st.button("Generate Executive Summary"):
                with st.spinner("Consulting AI Model..."):
                    report_text = AINarrativeGenerator.generate_report(
                        target_prog, 
                        prog_score, 
                        "Accreditation Ready" if prog_score > 0.8 else "Needs Improvement", 
                        [row_vals[str_col], str_col], 
                        [row_vals[weak_col], weak_col]
                    )
                    st.markdown(report_text)

    # --- TAB 4: NATIONAL COMPLIANCE (Single Institute View) ---
    with tab4:
        st.subheader("🇪🇹 Compliance vs. National Targets (MOLS)")
        targets = fetch_national_targets()
        
        # VISUAL 1: COC Pass Rate vs National Target
        st.markdown("#### 1. National COC (Certificate of Competency) Performance")
        
        fig_coc = go.Figure()
        fig_coc.add_trace(go.Bar(
            x=df.index, 
            y=df['COC_Pass_Rate'],
            name='Dept Pass Rate',
            marker_color='indianred'
        ))
        fig_coc.add_trace(go.Scatter(
            x=df.index, 
            y=[targets['Min_COC_Pass_Rate']] * len(df),
            mode='lines',
            name='National Target (85%)',
            line=dict(color='green', width=4, dash='dash')
        ))
        st.plotly_chart(fig_coc, use_container_width=True)

        # VISUAL 2: Kaizen & Coop Training
        st.markdown("#### 2. Kaizen Audit & Cooperative Training")
        col_nat1, col_nat2 = st.columns(2)
        
        with col_nat1:
            fig_kai = px.bar(
                df, x=df.index, y='Kaizen_Audit', 
                title="Kaizen 5S Implementation Score",
                color='Kaizen_Audit', color_continuous_scale='Blues'
            )
            fig_kai.add_hline(y=targets['Target_Kaizen_Score'], line_dash="dot", annotation_text="Target 90")
            st.plotly_chart(fig_kai, use_container_width=True)
            
        with col_nat2:
            fig_coop = px.scatter(
                df, x='Coop_Training_Rate', y='COC_Pass_Rate', 
                size='Unit_Cost', color=df.index,
                title="Strategy Map: Coop Training vs. Competency",
                labels={'Coop_Training_Rate': 'Cooperative Training (%)', 'COC_Pass_Rate': 'COC Pass (%)'}
            )
            st.plotly_chart(fig_coop, use_container_width=True)
            
        st.success("✅ **Governance Note:** All departments are required to upload their Kaizen Audit Checklist (Form K-01) to the central server monthly.")

if __name__ == "__main__":
    main()
