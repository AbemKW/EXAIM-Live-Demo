import asyncio

from demos.cdss_example.cdss import CDSS
from demos.cdss_example.schema.clinical_case import ClinicalCase, VitalSigns, LabResult


def format_summary_display(summary) -> str:
    """Format an AgentSummary for clean console display"""
    lines = [
        f"┌─ Status / Action: {summary.status_action}",
        f"├─ Key Findings: {summary.key_findings}",
        f"├─ Differential & Rationale: {summary.differential_rationale}",
        f"├─ Uncertainty / Confidence: {summary.uncertainty_confidence}",
        f"├─ Recommendation / Next Step: {summary.recommendation_next_step}",
        f"└─ Agent Contributions: {summary.agent_contributions}",
    ]
    return "\n".join(lines)


async def demo_case_1():
    """Demo Case 1: Chest Pain with Elevated Troponin"""
    print("\n" + "="*80)
    print("DEMO CASE 1: Chest Pain with Elevated Troponin")
    print("="*80)
    
    cdss = CDSS()
    
    case = ClinicalCase(
        patient_id="DEMO-001",
        age=58,
        sex="M",
        chief_complaint="Chest pain and shortness of breath",
        history_of_present_illness=(
            "58-year-old male presents with 2-hour history of substernal chest pain "
            "radiating to left arm, associated with diaphoresis and dyspnea. Pain started "
            "at rest, described as pressure-like, 7/10 severity. No relief with rest."
        ),
        past_medical_history=["Hypertension", "Type 2 Diabetes", "Hyperlipidemia"],
        medications=["Metformin 1000mg BID", "Lisinopril 10mg daily", "Atorvastatin 20mg daily"],
        allergies=["Penicillin"],
        vital_signs=VitalSigns(
            systolic_bp=145,
            diastolic_bp=92,
            heart_rate=98,
            respiratory_rate=20,
            temperature=37.1,
            oxygen_saturation=96
        ),
        lab_results=[
            LabResult(
                test_name="Troponin I",
                value=2.8,
                unit="ng/mL",
                reference_range="<0.04 ng/mL",
                status="critical",
                date="2025-11-09"
            ),
            LabResult(
                test_name="CK-MB",
                value=45,
                unit="ng/mL",
                reference_range="<5 ng/mL",
                status="high",
                date="2025-11-09"
            ),
            LabResult(
                test_name="BNP",
                value=850,
                unit="pg/mL",
                reference_range="<100 pg/mL",
                status="high",
                date="2025-11-09"
            ),
            LabResult(
                test_name="Creatinine",
                value=1.2,
                unit="mg/dL",
                reference_range="0.7-1.3 mg/dL",
                status="normal",
                date="2025-11-09"
            ),
            LabResult(
                test_name="Glucose",
                value=180,
                unit="mg/dL",
                reference_range="70-100 mg/dL",
                status="high",
                date="2025-11-09"
            )
        ],
        physical_exam=(
            "Alert, anxious, diaphoretic. Heart: regular rhythm, no murmurs. "
            "Lungs: clear bilaterally. Extremities: no edema."
        )
    )
    
    print("\nProcessing case through multi-agent CDSS...")
    print("-" * 80)
    
    result = await cdss.process_case(case)
    
    print("\n" + "="*80)
    print("AGENT SUMMARIES")
    print("="*80)
    
    if not result["agent_summaries"]:
        print("\n⚠️  No summaries were generated. This may indicate:")
        print("   - The BufferAgent did not trigger summarization")
        print("   - The reasoning streams were too short")
        print("   - Check debug output above for details")
    else:
        for i, summary_data in enumerate(result["agent_summaries"], 1):
            print(f"\nSummary {i}:")
            print(f"  Agents: {', '.join(summary_data['agents'])}")
            print(f"  Action: {summary_data['action']}")
            print(f"  Reasoning: {summary_data['reasoning']}")
            if summary_data['findings']:
                print(f"  Findings: {summary_data['findings']}")
            if summary_data['next_steps']:
                print(f"  Next Steps: {summary_data['next_steps']}")
    
    print("\n" + "="*80)
    print("FINAL RECOMMENDATION")
    print("="*80)
    final = result["final_recommendation"]
    print(f"\nAgents: {', '.join(final['agents'])}")
    print(f"Action: {final['action']}")
    print(f"Reasoning: {final['reasoning']}")
    if final['findings']:
        print(f"Findings: {final['findings']}")
    if final['next_steps']:
        print(f"Next Steps: {final['next_steps']}")
    
    print("\n" + "="*80)
    print("TRACE STATISTICS")
    print("="*80)
    print(f"Orchestrator traces: {result['trace_count']['orchestrator']}")
    print(f"Cardiology traces: {result['trace_count']['cardiology']}")
    print(f"Laboratory traces: {result['trace_count']['laboratory']}")


async def demo_case_2():
    """Demo Case 2: Free-text Case Description"""
    print("\n" + "="*80)
    print("DEMO CASE 2: Free-text Case - Heart Failure Exacerbation")
    print("="*80)
    
    cdss = CDSS()
    
    case_text = """
    Patient: 72-year-old female
    
    Chief Complaint: Progressive shortness of breath and leg swelling over 1 week
    
    History: Known heart failure with reduced ejection fraction (HFrEF), EF 35% from 2023.
    On furosemide 40mg daily, carvedilol 12.5mg BID, lisinopril 10mg daily.
    Recently missed several doses due to cost concerns.
    
    Vital Signs:
    - BP: 168/95 mmHg
    - HR: 110 bpm, irregular
    - RR: 24/min
    - SpO2: 92% on room air
    - Weight: Increased 8 lbs in 1 week
    
    Laboratory Results:
    - BNP: 1,200 pg/mL (elevated)
    - Creatinine: 1.8 mg/dL (elevated from baseline 1.1)
    - Sodium: 132 mEq/L (low)
    - Potassium: 3.2 mEq/L (low)
    - Troponin: 0.08 ng/mL (slightly elevated)
    
    Physical Exam:
    - JVD present at 45 degrees
    - Bibasilar rales
    - S3 gallop
    - 2+ pitting edema bilaterally
    - Hepatomegaly
    
    ECG: Atrial fibrillation with rapid ventricular response, rate 110 bpm
    """
    
    print("\nProcessing free-text case through multi-agent CDSS...")
    print("-" * 80)
    
    result = await cdss.process_case(case_text)
    
    print("\n" + "="*80)
    print("AGENT SUMMARIES")
    print("="*80)
    
    if not result["agent_summaries"]:
        print("\n⚠️  No summaries were generated. This may indicate:")
        print("   - The BufferAgent did not trigger summarization")
        print("   - The reasoning streams were too short")
        print("   - Check debug output above for details")
    else:
        for i, summary_data in enumerate(result["agent_summaries"], 1):
            print(f"\nSummary {i}:")
            print(f"  Agents: {', '.join(summary_data['agents'])}")
            print(f"  Action: {summary_data['action']}")
            print(f"  Reasoning: {summary_data['reasoning']}")
            if summary_data['findings']:
                print(f"  Findings: {summary_data['findings']}")
            if summary_data['next_steps']:
                print(f"  Next Steps: {summary_data['next_steps']}")
    
    print("\n" + "="*80)
    print("FINAL RECOMMENDATION")
    print("="*80)
    final = result["final_recommendation"]
    print(f"\nAgents: {', '.join(final['agents'])}")
    print(f"Action: {final['action']}")
    print(f"Reasoning: {final['reasoning']}")
    if final['findings']:
        print(f"Findings: {final['findings']}")
    if final['next_steps']:
        print(f"Next Steps: {final['next_steps']}")


async def demo_case_3():
    """Demo Case 3: Acute Coronary Syndrome"""
    print("\n" + "="*80)
    print("DEMO CASE 3: Acute Coronary Syndrome - Structured Case")
    print("="*80)
    
    cdss = CDSS()
    
    case = ClinicalCase(
        patient_id="DEMO-003",
        age=65,
        sex="F",
        chief_complaint="Acute onset chest pressure",
        history_of_present_illness=(
            "65-year-old female with sudden onset of severe substernal chest pressure "
            "30 minutes ago while watching TV. Pain radiates to jaw and both arms. "
            "Associated with nausea and lightheadedness. Denies dyspnea."
        ),
        past_medical_history=["Hypertension", "Obesity", "Family history of CAD"],
        medications=["Amlodipine 5mg daily"],
        allergies=[],
        vital_signs=VitalSigns(
            systolic_bp=110,
            diastolic_bp=70,
            heart_rate=55,
            respiratory_rate=18,
            temperature=36.8,
            oxygen_saturation=98
        ),
        lab_results=[
            LabResult(
                test_name="Troponin I",
                value=0.15,
                unit="ng/mL",
                reference_range="<0.04 ng/mL",
                status="high",
                date="2025-11-09"
            ),
            LabResult(
                test_name="CK-MB",
                value=8,
                unit="ng/mL",
                reference_range="<5 ng/mL",
                status="high",
                date="2025-11-09"
            ),
            LabResult(
                test_name="BNP",
                value=120,
                unit="pg/mL",
                reference_range="<100 pg/mL",
                status="slightly elevated",
                date="2025-11-09"
            ),
            LabResult(
                test_name="Creatinine",
                value=0.9,
                unit="mg/dL",
                reference_range="0.7-1.3 mg/dL",
                status="normal",
                date="2025-11-09"
            )
        ],
        physical_exam=(
            "Pale, diaphoretic. Heart: bradycardic, regular rhythm, no murmurs. "
            "Lungs: clear. Abdomen: soft, non-tender."
        ),
        imaging_results=["ECG: Sinus bradycardia, ST elevation in leads II, III, aVF"]
    )
    
    print("\nProcessing case through multi-agent CDSS...")
    print("-" * 80)
    
    result = await cdss.process_case(case)
    
    print("\n" + "="*80)
    print("AGENT SUMMARIES")
    print("="*80)
    
    if not result["agent_summaries"]:
        print("\n⚠️  No summaries were generated. This may indicate:")
        print("   - The BufferAgent did not trigger summarization")
        print("   - The reasoning streams were too short")
        print("   - Check debug output above for details")
    else:
        for i, summary_data in enumerate(result["agent_summaries"], 1):
            print(f"\nSummary {i}:")
            print(f"  Agents: {', '.join(summary_data['agents'])}")
            print(f"  Action: {summary_data['action']}")
            print(f"  Reasoning: {summary_data['reasoning']}")
            if summary_data['findings']:
                print(f"  Findings: {summary_data['findings']}")
            if summary_data['next_steps']:
                print(f"  Next Steps: {summary_data['next_steps']}")
    
    print("\n" + "="*80)
    print("FINAL RECOMMENDATION")
    print("="*80)
    final = result["final_recommendation"]
    print(f"\nAgents: {', '.join(final['agents'])}")
    print(f"Action: {final['action']}")
    print(f"Reasoning: {final['reasoning']}")
    if final['findings']:
        print(f"Findings: {final['findings']}")
    if final['next_steps']:
        print(f"Next Steps: {final['next_steps']}")


async def main():
    """Run all demo cases"""
    print("="*80)
    print("MULTI-AGENT CLINICAL DECISION SUPPORT SYSTEM (CDSS) DEMO")
    print("="*80)
    print("\nThis demo showcases a production-grade CDSS with three reasoning agents:")
    print("  - OrchestratorAgent: Coordinates workflow and synthesizes findings")
    print("  - CardiologyAgent: Cardiac assessment and recommendations")
    print("  - LaboratoryAgent: Lab result interpretation")
    print("\nAll agents use EXAIM for trace capture and summarization.")
    print("\n[STREAMING ENABLED] Agents stream tokens in real-time as they reason.")
    print("="*80)
    
    await demo_case_1()
    await demo_case_2()
    await demo_case_3()
    
    print("\n" + "="*80)
    print("DEMO COMPLETE")
    print("="*80)


if __name__ == "__main__":
    asyncio.run(main())

