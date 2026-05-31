import os
import json
from typing import Dict, Any, List
from google import genai
from google.genai import types

# Import ML Ingestion layers
from ml_layer.disaster_models import (
    simulate_weather_ingestion,
    forecast_weather_metrics,
    DisasterClassifier
)

# Import Context Tools
from tools.context_tools import fetch_news_context, get_live_weather_summary

# Import Insight Memory Manager
from memory.insight_manager import load_insights, save_insight

# Import Department Planners
from agents.department_agents import (
    generate_emergency_response_plan,
    generate_civil_defense_plan,
    generate_public_works_plan
)

# Import Cognitive Router
from agents.cognitive_router import run_cognitive_router

# State definition
class SystemState:
    """
    Tracks the state of the Autonomous Disaster Management System pipeline.
    """
    def __init__(self, location: str):
        self.location: str = location
        self.ml_prediction: dict = {}
        self.news_context: str = ""
        self.weather_context: str = ""
        self.target_department: str = ""
        self.routing_justification: str = ""
        self.drafted_plan: dict = {}
        self.human_decision: str = "Pending"  # Options: "Pending", "Approve", "Reject"
        self.human_feedback: str = ""
        self.loop_count: int = 0
        self.past_insights: List[str] = []

    def to_dict(self) -> dict:
        return {
            "location": self.location,
            "ml_prediction": self.ml_prediction,
            "news_context": self.news_context,
            "weather_context": self.weather_context,
            "target_department": self.target_department,
            "routing_justification": self.routing_justification,
            "drafted_plan": self.drafted_plan,
            "human_decision": self.human_decision,
            "human_feedback": self.human_feedback,
            "loop_count": self.loop_count,
            "past_insights": self.past_insights
        }

# Graph Node Definitions
def ingest_and_predict_node(state: SystemState) -> SystemState:
    """
    Ingests weather metrics, runs the classification model, and retrieves
    real-time weather and news context for the location.
    Defaults to scenario='heavy_rain' to simulate a severe event.
    """
    print(f"\n--- [1] Executing Ingestion & ML Prediction Node (Pass {state.loop_count + 1}) ---")
    
    # 1. Ingest simulated weather (force heavy_rain to trigger flood disaster testing)
    history_df = simulate_weather_ingestion(hours=168, scenario="heavy_rain", seed=42)
    forecast_df = forecast_weather_metrics(history_df, forecast_hours=48)
    
    # 2. Run prediction model
    classifier = DisasterClassifier()
    state.ml_prediction = classifier.predict_disaster(forecast_df)
    
    # 3. Retrieve weather & news contexts
    state.news_context = fetch_news_context(state.location)
    state.weather_context = get_live_weather_summary(state.location)
    
    print(f"ML Classifier predicted disaster type: {state.ml_prediction['disaster_type']} "
          f"with {state.ml_prediction['probability']:.1%} probability.")
    print("News and weather context blocks successfully fetched.")
    return state

def cognitive_routing_node(state: SystemState) -> SystemState:
    """
    Loads permanent reflection insights, executes the Cognitive Router to resolve 
    which department should own the disaster response phase.
    """
    print(f"\n--- [2] Executing Cognitive Routing Node ---")
    
    # Load memory insights
    state.past_insights = load_insights()
    print(f"Loaded {len(state.past_insights)} permanent rules from reflection memory.")
    
    # Run cognitive router
    routing_result = run_cognitive_router(
        ml_prediction=state.ml_prediction,
        news_context=state.news_context,
        weather_context=state.weather_context,
        past_insights=state.past_insights
    )
    
    state.target_department = routing_result.get("target_department")
    state.routing_justification = routing_result.get("justification")
    
    print(f"Cognitive Router resolved ownership: [{state.target_department}]")
    print(f"Justification: {state.routing_justification}")
    return state

def department_planning_node(state: SystemState) -> SystemState:
    """
    Routes the payload to the selected Department Agent, incorporating any past
    self-improving rules to draft a detailed action plan.
    """
    print(f"\n--- [3] Executing Department Planning Node ---")
    
    # Build payload
    payload = {
        "location": state.location,
        "disaster_type": state.ml_prediction["disaster_type"],
        "probability": state.ml_prediction["probability"],
        "severity": state.ml_prediction["severity"],
        "metrics": state.ml_prediction["metrics"],
        "news_context": state.news_context,
        "weather_context": state.weather_context
    }
    
    # Map department string to generation function
    agent_mapping = {
        "Emergency Response": generate_emergency_response_plan,
        "Civil Defense": generate_civil_defense_plan,
        "Public Works": generate_public_works_plan
    }
    
    planner = agent_mapping.get(state.target_department)
    if not planner:
        raise ValueError(f"Unknown target department resolved: {state.target_department}")
        
    # Generate structured plan
    state.drafted_plan = planner(payload, past_insights=state.past_insights)
    
    print("Action plan and citizen alerts successfully generated.")
    return state

def human_gatekeeper_node(state: SystemState) -> SystemState:
    """
    Human-in-the-Loop checkpoint. Pauses execution, prints outputs to console,
    and prompts the user to Approve (A) or Reject (R) the plans.
    """
    print("\n" + "=" * 60)
    print(f"HUMAN-IN-THE-LOOP GATEKEEPER: REVIEW PLAN FOR {state.location.upper()}")
    print("=" * 60)
    print(f"Target Department: {state.target_department}")
    print(f"Routing Reason:    {state.routing_justification}")
    print("-" * 60)
    print(f"BROADCAST ALERT MESSAGE:\n  \"{state.drafted_plan.get('alert_message')}\"")
    print("\nACTION PLAN:")
    for idx, step in enumerate(state.drafted_plan.get("action_plan", []), 1):
        print(f"  {idx}. {step}")
    print(f"\nSeverity Verification:\n  {state.drafted_plan.get('severity_verification')}")
    print("=" * 60)
    
    # Interactive Console Prompt
    while True:
        choice = input("\nApprove Plan (A) or Reject & Provide Feedback (R)? ").strip().upper()
        if choice in ["A", "APPROVE"]:
            state.human_decision = "Approve"
            state.human_feedback = ""
            print("[INFO] Plan approved by human operator.")
            break
        elif choice in ["R", "REJECT"]:
            state.human_decision = "Reject"
            feedback = input("Enter your feedback/critique: ").strip()
            state.human_feedback = feedback
            print("[INFO] Plan rejected. Entering self-correction phase.")
            break
        else:
            print("[WARNING] Invalid choice. Please enter 'A' or 'R'.")
            
    return state

def self_improvement_node(state: SystemState) -> SystemState:
    """
    Critique compilation and model self-improvement.
    Transforms raw user feedback into a permanent structured engineering rule using Gemini.
    """
    print(f"\n--- [5] Executing Self-Improvement Memory Node ---")
    
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    
    if not api_key:
        print("[INFO] GEMINI_API_KEY missing. Generating deterministic reflection rule locally...")
        # Create a mock rule reflecting the user feedback directly
        new_rule = f"Rule: Always ensure that: {state.human_feedback}"
    else:
        # Prompt Gemini to write a structured, abstract, and generalized rule
        print("Consulting Reflection Model to abstract critique into a rule...")
        prompt = (
            f"A human disaster controller has REJECTED the following drafted action plan.\n\n"
            f"Target Department: {state.target_department}\n"
            f"Drafted Alert Message: {state.drafted_plan.get('alert_message')}\n"
            f"Drafted Action Plan:\n"
            f"{json.dumps(state.drafted_plan.get('action_plan'), indent=2)}\n\n"
            f"Human Critique/Feedback:\n"
            f"\"{state.human_feedback}\"\n\n"
            f"Identify the human's core concern. Formulate a single, concise, and generalized "
            f"engineering rule or guideline that starts with 'Rule: ' (e.g., 'Rule: Always specify shelter coordinates near central locations'). "
            f"This rule will be fed back into the prompt in subsequent runs so the model never repeats this mistake. "
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
        new_rule = response.text.strip()
        
    # Append rule to insights database
    save_insight(new_rule)
    print(f"\n>>> REFLECTION MEMORY UPDATED:")
    print(f"  Saved New Rule: \"{new_rule}\"")
    
    # Reset human decision to allow loop re-entry, increment count
    state.human_decision = "Pending"
    state.loop_count += 1
    
    return state

# Core Orchestration Graph Loop
def run_disaster_system(location: str):
    """
    Orchestrates the state transitions continuously until a plan is approved by the human.
    """
    # Initialize system state
    state = SystemState(location)
    
    while state.human_decision != "Approve":
        # Node 1: Weather simulation & forecast prediction
        state = ingest_and_predict_node(state)
        
        # Node 2: Cognitive Routing based on rules
        state = cognitive_routing_node(state)
        
        # Node 3: Department planning execution
        state = department_planning_node(state)
        
        # Node 4: Human-in-the-Loop gatekeeper checkpoint
        state = human_gatekeeper_node(state)
        
        # Node 5: Self-correction (only trigger if rejected)
        if state.human_decision == "Reject":
            state = self_improvement_node(state)
            
    print("\n" + "=" * 60)
    print(f"SUCCESS: System successfully exited after {state.loop_count} correction loop(s).")
    print(f"Final approved payload saved.")
    print("=" * 60)

if __name__ == "__main__":
    print("=" * 70)
    print("AUTONOMOUS DISASTER MANAGEMENT SYSTEM (ADMS) - RUNTIME ENGINE")
    print("=" * 70)
    
    # Wipe previous insights file if clean test is preferred
    insights_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "memory", "insights.json")
    if os.path.exists(insights_path):
        try:
            os.remove(insights_path)
            print("[INFO] Wiped memory/insights.json for a clean test run.")
        except IOError:
            pass
            
    test_location = "Jaipur"
    run_disaster_system(test_location)
