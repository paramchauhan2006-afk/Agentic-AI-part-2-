import streamlit as st
import os
import sys
import json
from google import genai
from google.genai import types

# Add parent directory to sys.path to enable imports of sibling packages (memory, ml_layer, etc.)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import project modules
from memory.insight_manager import load_insights, save_insight
from ml_layer.disaster_models import (
    simulate_weather_ingestion,
    forecast_weather_metrics,
    DisasterClassifier
)
from tools.context_tools import fetch_news_context, get_live_weather_summary
from agents.cognitive_router import run_cognitive_router
from agents.department_agents import (
    generate_emergency_response_plan,
    generate_civil_defense_plan,
    generate_public_works_plan
)

# Streamlit App Styling Configuration
st.set_page_config(
    page_title="ADMS Dashboard",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling rules injected to Streamlit
st.markdown("""
    <style>
    .metric-card {
        background-color: #0e1117;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #30363d;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    .broadcast-box {
        background-color: #1f1f2e;
        border-left: 5px solid #ff4b4b;
        padding: 1.2rem;
        border-radius: 5px;
        margin-bottom: 1.5rem;
        color: #ffcccc;
        font-weight: 500;
        font-family: monospace;
    }
    .routing-box {
        background-color: #1a2421;
        border: 1px solid #238636;
        padding: 1.2rem;
        border-radius: 8px;
        margin-bottom: 1.5rem;
        color: #e6f7ed;
    }
    </style>
""", unsafe_allow_html=True)

# Helper function to run the full pipeline in session state
def execute_pipeline(location: str, weather_profile: str):
    """
    Executes the ingestion, prediction, routing, and planning sequence,
    and updates the session state.
    """
    # Map selector profile to simulation scenario strings
    profile_mapping = {
        "NORMAL": "normal",
        "HEAVY_RAIN": "heavy_rain",
        "HIGH_WINDS": "high_winds",
        "EXTREME_HEAT": "extreme_heat"
    }
    scenario = profile_mapping.get(weather_profile, "normal")
    
    with st.spinner("Executing simulation layers (Ingestion -> prediction -> router)..."):
        # 1. Ingestion & Forecasting
        history_df = simulate_weather_ingestion(hours=168, scenario=scenario, seed=42)
        forecast_df = forecast_weather_metrics(history_df, forecast_hours=48)
        
        # 2. ML Classifier
        classifier = DisasterClassifier()
        ml_prediction = classifier.predict_disaster(forecast_df)
        
        # 3. Context Tools
        news = fetch_news_context(location)
        weather = get_live_weather_summary(location)
        
        # 4. Load memory insights
        past_insights = load_insights()
        
        # 5. Cognitive Router
        routing_result = run_cognitive_router(
            ml_prediction=ml_prediction,
            news_context=news,
            weather_context=weather,
            past_insights=past_insights
        )
        
        target_dept = routing_result.get("target_department")
        justification = routing_result.get("justification")
        
        # 6. Department Agent Planning
        agent_mapping = {
            "Emergency Response": generate_emergency_response_plan,
            "Civil Defense": generate_civil_defense_plan,
            "Public Works": generate_public_works_plan
        }
        planner = agent_mapping.get(target_dept)
        if not planner:
            planner = generate_emergency_response_plan
            
        payload = {
            "location": location,
            "disaster_type": ml_prediction["disaster_type"],
            "probability": ml_prediction["probability"],
            "severity": ml_prediction["severity"],
            "metrics": ml_prediction["metrics"],
            "news_context": news,
            "weather_context": weather
        }
        drafted_plan = planner(payload, past_insights=past_insights)
        
        # Save to state
        st.session_state.state = {
            "location": location,
            "weather_profile": weather_profile,
            "ml_prediction": ml_prediction,
            "news_context": news,
            "weather_context": weather,
            "target_department": target_dept,
            "routing_justification": justification,
            "drafted_plan": drafted_plan,
            "past_insights": past_insights
        }
        st.session_state.show_hil_panel = True
        st.session_state.approved = False
        st.session_state.rejection_active = False

def abstract_feedback_to_rule(feedback: str, state: dict) -> str:
    """
    Translates raw human critique into a structured permanent rule using Gemini
    or falls back to a deterministic format if GEMINI_API_KEY is not set.
    """
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        return f"Rule: Always ensure that: {feedback}"
        
    prompt = (
        f"A human disaster controller has REJECTED the following drafted action plan.\n\n"
        f"Target Department: {state.get('target_department')}\n"
        f"Drafted Alert Message: {state.get('drafted_plan', {}).get('alert_message')}\n"
        f"Drafted Action Plan:\n"
        f"{json.dumps(state.get('drafted_plan', {}).get('action_plan'), indent=2)}\n\n"
        f"Human Critique/Feedback:\n"
        f"\"{feedback}\"\n\n"
        f"Identify the human's core concern. Formulate a single, concise, and generalized "
        f"engineering rule or guideline that starts with 'Rule: ' (e.g., 'Rule: Always specify shelter coordinates near central locations'). "
        f"Return ONLY the rule string."
    )
    
    client = genai.Client()
    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.1
        )
    )
    return response.text.strip()

# Initialize Session State
if "state" not in st.session_state:
    st.session_state.state = None
if "show_hil_panel" not in st.session_state:
    st.session_state.show_hil_panel = False
if "approved" not in st.session_state:
    st.session_state.approved = False
if "rejection_active" not in st.session_state:
    st.session_state.rejection_active = False
if "feedback_input" not in st.session_state:
    st.session_state.feedback_input = ""

# ================= SIDEBAR: MEMORY PANEL =================
st.sidebar.title("🧠 System Memory & Learned Rules")
st.sidebar.markdown(
    "This list displays permanent rules of thumb and design preferences "
    "retrieved from previous human feedback loops. The system reads "
    "these insights to self-improve on subsequent runs."
)

insights = load_insights()
if not insights:
    st.sidebar.info("Memory is currently empty. Rejections in the Human-in-the-Loop review will populate this database.")
else:
    for idx, rule in enumerate(insights, 1):
        st.sidebar.markdown(f"**{idx}.** `{rule}`")

st.sidebar.markdown("---")
if st.sidebar.button("Wipe Insights Database", use_container_width=True):
    insights_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "memory", "insights.json")
    if os.path.exists(insights_path):
        try:
            with open(insights_path, "w", encoding="utf-8") as f:
                json.dump([], f)
            st.sidebar.success("Database cleared!")
            st.rerun()
        except IOError as e:
            st.sidebar.error(f"Failed to wipe: {e}")

# ================= MAIN CONTENT: SYSTEM PANEL =================
st.title("🚨 ADMS: Autonomous Disaster Management System")
st.markdown("State-driven cognitive orchestration mapping forecast data to department planning protocols.")

# 1. Main Controls Container
st.markdown("### ⚙️ Simulator Ingestion Controls")
with st.container(border=True):
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        loc_input = st.text_input("Target Geographic Location", value="Jaipur")
    with col2:
        profile_input = st.selectbox(
            "Simulated Meteorological Profile",
            options=["NORMAL", "HEAVY_RAIN", "HIGH_WINDS", "EXTREME_HEAT"]
        )
    with col3:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        trigger = st.button("Trigger Simulation Engine", use_container_width=True, type="primary")

if trigger:
    execute_pipeline(loc_input, profile_input)
    st.rerun()

# 2. Output Panel (only if state exists)
if st.session_state.state:
    state_data = st.session_state.state
    ml = state_data["ml_prediction"]
    
    st.markdown("### 📊 Ingested Analytics & Routing")
    
    # Metric cards for classifier output
    mcol1, mcol2, mcol3 = st.columns(3)
    with mcol1:
        st.markdown(
            f"<div class='metric-card'><h4>Disaster Target</h4><h2 style='color:#ff4b4b;'>{ml['disaster_type']}</h2></div>",
            unsafe_allow_html=True
        )
    with mcol2:
        st.markdown(
            f"<div class='metric-card'><h4>ML Probability</h4><h2 style='color:#e5c158;'>{ml['probability']:.1%}</h2></div>",
            unsafe_allow_html=True
        )
    with mcol3:
        st.markdown(
            f"<div class='metric-card'><h4>Assigned Severity</h4><h2 style='color:#ff4b4b;'>{ml['severity']}</h2></div>",
            unsafe_allow_html=True
        )
        
    st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)
    
    # Detail columns
    det_col1, det_col2 = st.columns(2)
    with det_col1:
        with st.expander("📡 Ingested Weather Station Observations", expanded=False):
            st.text(state_data["weather_context"])
        with st.expander("📰 Localized Municipal News Feed", expanded=False):
            st.text(state_data["news_context"])
            
    with det_col2:
        routing_html = (
            f"<div class='routing-box'>"
            f"<h4>⚡ Routing Ownership: {state_data['target_department']}</h4>"
            f"<p style='font-size:0.95rem; margin-top:0.5rem;'><strong>Justification:</strong> {state_data['routing_justification']}</p>"
            f"</div>"
        )
        st.markdown(routing_html, unsafe_allow_html=True)

    # 3. Human-in-the-Loop Review Panel
    if st.session_state.show_hil_panel:
        st.markdown("### 🛡️ Human-in-the-Loop Review Gate")
        plan = state_data["drafted_plan"]
        
        with st.container(border=True):
            st.subheader(f"Proposed Protocol: {state_data['target_department']} Plan")
            
            # Broadcast Message alert box
            st.markdown(
                f"<div class='broadcast-box'>📢 BROADCAST ALERT:<br/>{plan.get('alert_message')}</div>",
                unsafe_allow_html=True
            )
            
            # Action Steps list
            st.markdown("**Chronological Action Plan Tasks:**")
            for idx, task in enumerate(plan.get("action_plan", []), 1):
                st.markdown(f"**{idx}.** {task}")
                
            st.markdown("<br/>", unsafe_allow_html=True)
            st.markdown(f"**Severity Verification:** {plan.get('severity_verification')}")
            
            st.markdown("---")
            
            # Action Buttons
            btn_col1, btn_col2, _ = st.columns([1, 1, 3])
            with btn_col1:
                approve_clicked = st.button("🟢 Approve & Broadcast", use_container_width=True)
            with btn_col2:
                reject_clicked = st.button("🔴 Reject & Correct Plan", use_container_width=True)
                
            if approve_clicked:
                st.session_state.approved = True
                st.session_state.show_hil_panel = False
                st.session_state.rejection_active = False
                st.rerun()
                
            if reject_clicked:
                st.session_state.rejection_active = True
                st.rerun()
                
            # Rejection entry box
            if st.session_state.rejection_active:
                st.markdown("#### ⚙️ Self-Correction Rules Generator")
                critique_text = st.text_input("Enter human feedback/critique here:")
                
                if st.button("Submit Corrective Rule"):
                    if not critique_text.strip():
                        st.warning("Feedback field cannot be empty.")
                    else:
                        with st.spinner("Processing critique and updating insights..."):
                            # Abstract feedback to a rule
                            new_rule = abstract_feedback_to_rule(critique_text, state_data)
                            save_insight(new_rule)
                            
                            # Rerun pipeline immediately to apply rule
                            execute_pipeline(state_data["location"], state_data["weather_profile"])
                            st.success(f"Rule added: \"{new_rule}\". Plan optimized!")
                            st.rerun()

# 4. Final success alert if approved
if st.session_state.approved:
    st.success("🟢 Alert broadcasted successfully. Execution finalized.")
    st.balloons()
