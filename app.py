import gradio as gr
import asyncio
import threading
import time
import sys
from pathlib import Path

# Add project paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from demos.cdss_example.cdss import CDSS
from exaim_core.schema.agent_summary import AgentSummary


class GradioStreamingHandler:
    """Handler for streaming agent traces and summaries to Gradio interface"""
    
    def __init__(self):
        self.raw_traces = {}  # agent_id -> list of trace text
        self.summaries = []
        self.current_agent = None
        self.processing = False
        self.error = None
        self.agent_order = []  # Track order of agent appearances
        
    def reset(self):
        """Reset handler state for new case"""
        self.raw_traces = {}
        self.summaries = []
        self.current_agent = None
        self.processing = False
        self.error = None
        self.agent_order = []
    
    def trace_callback(self, agent_id: str, token: str):
        """Callback for receiving trace tokens"""
        if agent_id not in self.raw_traces:
            self.raw_traces[agent_id] = []
            self.agent_order.append(agent_id)
            self.current_agent = agent_id
        
        self.raw_traces[agent_id].append(token)
    
    def summary_callback(self, summary: AgentSummary):
        """Callback for receiving summaries"""
        print(f"🎯 GRADIO CALLBACK: Received summary #{len(self.summaries) + 1}")
        if summary and hasattr(summary, 'status_action'):
            print(f"   Status: {summary.status_action[:50] if len(summary.status_action) > 50 else summary.status_action}")
            print(f"   Key Findings: {summary.key_findings}")
        else:
            print(f"   Warning: Summary object is {summary}")
        self.summaries.append(summary)
    
    def get_agent_outputs(self) -> dict:
        """Get individual agent outputs as dictionary"""
        outputs = {}
        for agent_id in self.agent_order:
            if agent_id in self.raw_traces:
                outputs[agent_id] = "".join(self.raw_traces[agent_id])
        return outputs
    
    def format_raw_traces(self) -> str:
        """Format raw traces for display with custom styling"""
        if not self.raw_traces:
            return "⏳ Waiting for agent activity..."
        
        output = []
        
        # Agent emoji mapping
        agent_emojis = {
            "OrchestratorAgent": "🎯",
            "CardiologyAgent": "❤️",
            "NeurologyAgent": "🧠",
            "InfectiousDiseaseAgent": "🦠",
            "InternalMedicineAgent": "🏥",
            "LaboratoryAgent": "🔬",
            "RadiologyAgent": "📸",
            "SurgeryAgent": "⚕️",
            "PediatricsAgent": "👶",
        }
        
        for agent_id in self.agent_order:
            if agent_id not in self.raw_traces:
                continue
                
            traces = self.raw_traces[agent_id]
            emoji = agent_emojis.get(agent_id, "🤖")
            
            # Create styled card for each agent
            output.append(f"\n---\n\n")
            output.append(f"### {emoji} **{agent_id}**\n\n")
            
            # Add status indicator
            if agent_id == self.current_agent and self.processing:
                output.append("🟢 *Currently active...*\n\n")
            else:
                output.append("✅ *Completed*\n\n")
            
            # Add content with word wrapping (use blockquote instead of code block)
            content = "".join(traces)
            if content:
                # Split long lines and format for better wrapping
                lines = content.split('\n')
                for line in lines:
                    # Wrap very long lines
                    if len(line) > 80:
                        words = line.split(' ')
                        current_line = []
                        current_length = 0
                        for word in words:
                            if current_length + len(word) + 1 > 80 and current_line:
                                output.append(' '.join(current_line) + '\n')
                                current_line = [word]
                                current_length = len(word)
                            else:
                                current_line.append(word)
                                current_length += len(word) + 1
                        if current_line:
                            output.append(' '.join(current_line) + '\n')
                    else:
                        output.append(line + '\n')
                output.append("\n")
        
        return "".join(output)
    
    def format_latest_summary(self) -> str:
        """Format the most recent summary with custom styling"""
        if not self.summaries:
            return "⏳ **Waiting for summaries...**\n\n*EXAIM will generate summaries as agents complete their reasoning.*"
        
        latest = self.summaries[-1]
        output = []
        
        # Header with summary count
        output.append(f"### 📊 Summary #{len(self.summaries)}\n\n")
        
        # Status/Action
        if latest.status_action:
            output.append(f"**Status**: {latest.status_action}\n\n")
        
        # Key findings
        if latest.key_findings:
            output.append(f"**🔑 Key Findings**: {latest.key_findings}\n\n")
        
        # Differential & Rationale
        if latest.differential_rationale:
            output.append(f"**🧬 Differential & Rationale**: {latest.differential_rationale}\n\n")
        
        # Uncertainty/Confidence
        if latest.uncertainty_confidence:
            output.append(f"**📊 Uncertainty/Confidence**: {latest.uncertainty_confidence}\n\n")
        
        # Recommendations
        if latest.recommendation_next_step:
            output.append(f"**💡 Next Step**: {latest.recommendation_next_step}\n\n")
        
        # Agent Contributions
        if latest.agent_contributions:
            output.append(f"**🤖 Agent Contributions**: {latest.agent_contributions}\n\n")
        
        return "".join(output)
    
    def format_carousel_summaries(self) -> str:
        """Format all summaries in a scrollable carousel view"""
        if not self.summaries:
            return "⏳ **No summaries yet**\n\nSummaries will appear here as they are generated."
        
        output = []
        
        for idx, summary in enumerate(self.summaries, 1):
            output.append(f"\n---\n\n")
            output.append(f"### 📊 Summary #{idx}\n\n")
            
            # Compact view - just key findings (first 80 chars)
            if summary.key_findings:
                findings_short = summary.key_findings[:80] + "..." if len(summary.key_findings) > 80 else summary.key_findings
                output.append(f"**Key Findings**: {findings_short}\n\n")
            
            # Status
            if summary.status_action:
                status_short = summary.status_action[:60] + "..." if len(summary.status_action) > 60 else summary.status_action
                output.append(f"*{status_short}*\n\n")
        
        return "".join(output)


# Global streaming handler
streaming_handler = GradioStreamingHandler()


def process_case_gradio(case_text: str):
    """
    Process a clinical case and yield streaming updates for Gradio
    """
    if not case_text or not case_text.strip():
        yield (
            "⚠️ **No case provided**\n\nPlease enter a clinical case to analyze.",
            "⚠️ **No case provided**\n\nPlease enter a clinical case to analyze.",
            "⚠️ **No case provided**"
        )
        return
    
    # Reset handler
    streaming_handler.reset()
    streaming_handler.processing = True
    
    # Initial state
    yield (
        "🔄 **Initializing...**\n\nSetting up multi-agent system...",
        "🔄 **Initializing...**\n\nWaiting for first summary...",
        "⏳ **Processing...**"
    )
    
    # Flag to track if we're done
    processing_complete = False
    
    async def async_processor():
        """Run the CDSS asynchronously"""
        nonlocal processing_complete
        try:
            print("🚀 Starting CDSS analysis...")
            
            # Initialize CDSS
            cdss = CDSS()
            
            # Register callbacks with EXAIM
            cdss.exaim.register_trace_callback(streaming_handler.trace_callback)
            cdss.exaim.register_summary_callback(streaming_handler.summary_callback)
            
            # Run the analysis asynchronously
            result = await cdss.process_case(case_text)
            
            print(f"✅ CDSS completed with {len(streaming_handler.summaries)} summaries")
            
        except Exception as e:
            print(f"❌ Error in background processor: {str(e)}")
            import traceback
            traceback.print_exc()
            streaming_handler.error = str(e)
        finally:
            streaming_handler.processing = False
            processing_complete = True
    
    def background_processor():
        """Run the async processor in a background thread with asyncio"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(async_processor())
        finally:
            loop.close()
    
    # Start background thread
    thread = threading.Thread(target=background_processor, daemon=True)
    thread.start()
    
    # Stream updates while processing
    last_trace_update = ""
    last_summary_update = ""
    last_carousel_update = ""
    update_count = 0
    
    while not processing_complete or thread.is_alive():
        # Check for errors
        if streaming_handler.error:
            yield (
                f"❌ **Error occurred**\n\n```\n{streaming_handler.error}\n```",
                f"❌ **Error occurred**\n\n```\n{streaming_handler.error}\n```",
                "❌ **Error**"
            )
            return
        
        # Get current state
        current_trace = streaming_handler.format_raw_traces()
        current_summary = streaming_handler.format_latest_summary()
        current_carousel = streaming_handler.format_carousel_summaries()
        
        # Only yield if something changed
        if (current_trace != last_trace_update or 
            current_summary != last_summary_update or
            current_carousel != last_carousel_update):
            
            last_trace_update = current_trace
            last_summary_update = current_summary
            last_carousel_update = current_carousel
            update_count += 1
            
            yield (current_trace, current_summary, current_carousel)
        
        # Small delay to avoid overwhelming the UI
        time.sleep(0.5)
        
        # Safety timeout
        if update_count > 200:  # ~100 seconds max
            print("⚠️ Timeout reached, stopping updates")
            break
    
    # Final update
    streaming_handler.processing = False
    final_trace = streaming_handler.format_raw_traces()
    final_summary = streaming_handler.format_latest_summary()
    final_carousel = streaming_handler.format_carousel_summaries()
    
    print(f"🏁 Final state: {len(streaming_handler.summaries)} summaries, {len(streaming_handler.agent_order)} agents")
    
    yield (final_trace, final_summary, final_carousel)


# Example clinical cases
EXAMPLE_CASES = [
    """Patient: 68-year-old male
Chief Complaint: Chest pain and shortness of breath for 2 hours

HPI: Patient reports sudden onset of crushing substernal chest pain radiating to left arm and jaw. Associated with diaphoresis and nausea. Pain started while climbing stairs. No relief with rest.

PMH: Hypertension, type 2 diabetes, hyperlipidemia
Medications: Metformin, Lisinopril, Atorvastatin
Social: 30 pack-year smoking history, quit 5 years ago

Vitals:
- BP: 168/95 mmHg
- HR: 105 bpm, regular
- RR: 22 breaths/min
- Temp: 37.1°C
- SpO2: 94% on room air

Physical Exam:
- General: Anxious, diaphoretic
- Cardiovascular: Tachycardic, regular rhythm, no murmurs
- Respiratory: Mild tachypnea, clear breath sounds bilaterally
- Extremities: No edema, peripheral pulses intact""",

    """Patient: 45-year-old female
Chief Complaint: Severe headache and fever for 3 days

HPI: Progressive headache, now 10/10 in severity. Photophobia and neck stiffness developed yesterday. Fever up to 39.2°C. One episode of projectile vomiting this morning. No recent trauma.

PMH: Unremarkable
Medications: Oral contraceptives
Social: Works as elementary school teacher

Vitals:
- BP: 132/78 mmHg
- HR: 98 bpm
- RR: 18 breaths/min
- Temp: 39.0°C
- SpO2: 98% on room air

Physical Exam:
- General: Ill-appearing, photophobic
- HEENT: Pupils equal and reactive, no papilledema on fundoscopy
- Neck: Stiff with positive Kernig's and Brudzinski's signs
- Neurological: Alert and oriented x3, no focal deficits
- Skin: No rashes noted""",

    """Patient: 72-year-old female
Chief Complaint: Progressive weakness and confusion for 1 week

HPI: Family reports patient has become increasingly lethargic. Not eating well. Multiple episodes of vomiting. Decreased urine output noticed. No fever or cough. Recently started new medication for arthritis.

PMH: Osteoarthritis, chronic kidney disease (Stage 3), hypothyroidism
Medications: Levothyroxine, Recently started NSAIDs for arthritis pain
Social: Lives alone, independent in ADLs until this week

Vitals:
- BP: 88/52 mmHg (baseline 130/80)
- HR: 58 bpm
- RR: 16 breaths/min
- Temp: 36.8°C
- SpO2: 97% on room air

Physical Exam:
- General: Lethargic, slow to respond
- Skin: Cool, dry, decreased turgor
- Cardiovascular: Bradycardic, regular rhythm
- Abdomen: Soft, non-tender, hypoactive bowel sounds
- Neurological: Oriented to person only, slow mentation"""
]


# Custom CSS for better styling and visibility
custom_css = """
/* Fixed height containers with equal sizing */
#raw_output, #summary_output {
    min-height: 600px !important;
    max-height: 600px !important;
    height: 600px !important;
    overflow-y: auto !important;
    overflow-x: hidden !important;
    padding: 20px !important;
    border-radius: 8px !important;
    background: #1a1a1a !important;
    color: #e0e0e0 !important;
    border: 1px solid #333 !important;
}

#carousel_output {
    min-height: 300px !important;
    max-height: 300px !important;
    height: 300px !important;
    overflow-y: auto !important;
    overflow-x: hidden !important;
    padding: 20px !important;
    border-radius: 8px !important;
    background: #1a1a1a !important;
    color: #e0e0e0 !important;
    border: 1px solid #333 !important;
}

/* Ensure parent containers don't flex */
#raw_output > div,
#summary_output > div,
#carousel_output > div {
    height: 100% !important;
}

.markdown-content {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
    line-height: 1.6;
    color: #e0e0e0 !important;
}

.markdown-content h3 {
    color: #4fc3f7 !important;
    margin-top: 1em;
    margin-bottom: 0.5em;
}

.markdown-content strong {
    color: #81c784 !important;
}

.markdown-content code {
    background-color: #2d2d2d !important;
    color: #ff9800 !important;
    padding: 2px 6px;
    border-radius: 3px;
    font-size: 0.9em;
}

.markdown-content p {
    color: #e0e0e0 !important;
}

.markdown-content em {
    color: #b0b0b0 !important;
}

/* Fix for markdown content inside outputs */
#raw_output .markdown-content,
#summary_output .markdown-content,
#carousel_output .markdown-content {
    color: #e0e0e0 !important;
}

#raw_output h3,
#summary_output h3,
#carousel_output h3 {
    color: #4fc3f7 !important;
}

#raw_output p,
#summary_output p,
#carousel_output p {
    color: #e0e0e0 !important;
}

/* Scrollbar styling for dark theme */
#raw_output::-webkit-scrollbar,
#summary_output::-webkit-scrollbar,
#carousel_output::-webkit-scrollbar {
    width: 8px;
}

#raw_output::-webkit-scrollbar-track,
#summary_output::-webkit-scrollbar-track,
#carousel_output::-webkit-scrollbar-track {
    background: #2d2d2d;
}

#raw_output::-webkit-scrollbar-thumb,
#summary_output::-webkit-scrollbar-thumb,
#carousel_output::-webkit-scrollbar-thumb {
    background: #555;
    border-radius: 4px;
}

#raw_output::-webkit-scrollbar-thumb:hover,
#summary_output::-webkit-scrollbar-thumb:hover,
#carousel_output::-webkit-scrollbar-thumb:hover {
    background: #777;
}
"""

# JavaScript for auto-scrolling the Raw MAS output
auto_scroll_js = """
<script>
function autoScrollRawOutput() {
    const rawOutput = document.getElementById('raw_output');
    if (rawOutput) {
        // Scroll to bottom smoothly
        rawOutput.scrollTop = rawOutput.scrollHeight;
    }
}

// Create a MutationObserver to watch for changes in raw_output
const observer = new MutationObserver((mutations) => {
    mutations.forEach((mutation) => {
        if (mutation.target.id === 'raw_output' || mutation.target.closest('#raw_output')) {
            autoScrollRawOutput();
        }
    });
});

// Start observing when the page loads
window.addEventListener('load', () => {
    const rawOutput = document.getElementById('raw_output');
    if (rawOutput) {
        observer.observe(rawOutput, {
            childList: true,
            subtree: true,
            characterData: true
        });
    }
});
</script>
"""

with gr.Blocks(
    title="EXAIM - Clinical Decision Support Demo",
    theme=gr.themes.Default(primary_hue="blue", secondary_hue="cyan"),
    css=custom_css,
    head=auto_scroll_js
) as demo:
    
    gr.Markdown("""
    # 🏥 EXAIM: Explainable AI Middleware
    
    This demo showcases EXAIM's ability to compress and summarize multi-agent clinical reasoning in real-time.
    
    ### How it works:
    1. **Enter a patient case** in the text box below
    2. **Click "Analyze Case"** to process through our multi-agent system
    3. **Compare outputs**: Raw agent traces (left) vs. EXAIM summaries (right)
    
    EXAIM automatically identifies key clinical insights and removes redundant reasoning, providing clinicians 
    with actionable summaries without losing critical information.
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📝 Patient Case Input")
            
            case_input = gr.Textbox(
                label="Clinical Case Description",
                placeholder="Enter patient presentation, history, vitals, and physical exam findings...",
                lines=12,
                max_lines=20
            )
            
            submit_btn = gr.Button("🔬 Analyze Case", variant="primary", size="lg")
            
            gr.Markdown("### 📋 Example Cases")
            gr.Markdown("*Click an example below to load it:*")
            
            example_buttons = []
            for idx, example in enumerate(EXAMPLE_CASES, 1):
                btn = gr.Button(f"Example {idx}", size="sm")
                example_buttons.append((btn, example))
    
    gr.Markdown("---")
    gr.Markdown("## 🔍 Analysis Results")
    
    with gr.Row(equal_height=True):
        with gr.Column(scale=1):
            gr.Markdown("### 🤖 Multi-Agent System - Live Reasoning")
            gr.Markdown("*Watch each specialist agent contribute their expertise in real-time*")
            raw_output = gr.Markdown(
                value="⏳ **Waiting for case input...**\n\nAgent reasoning will appear here as cards, showing each specialist's analysis.",
                container=True,
                elem_id="raw_output",
                line_breaks=True,
                elem_classes=["markdown-content"]
            )
        
        with gr.Column(scale=1):
            gr.Markdown("### ✨ EXAIM Latest Summary")
            gr.Markdown("*Most recent compressed clinical insight*")
            summary_output = gr.Markdown(
                value="⏳ **Waiting for summaries...**\n\n*EXAIM will generate summaries as agents complete their reasoning.*",
                container=True,
                elem_id="summary_output",
                line_breaks=True,
                elem_classes=["markdown-content"]
            )
    
    gr.Markdown("---")
    gr.Markdown("## 📜 Summary Timeline")
    gr.Markdown("*Quick overview of all summaries - scroll to see more*")
    
    with gr.Row():
        with gr.Column(scale=1):
            carousel_output = gr.Markdown(
                value="⏳ **No summaries yet**\n\nSummaries will appear here as they are generated.",
                container=True,
                elem_id="carousel_output",
                line_breaks=True,
                elem_classes=["markdown-content"]
            )
    
    gr.Markdown("""
    ---
    ### 📊 What makes EXAIM different?
    
    - **Intelligent Compression**: Removes redundancy while preserving critical information
    - **Real-time Summarization**: Generates summaries as agents complete their reasoning
    - **Clinically-Focused**: Extracts key findings, differentials, and recommendations
    - **Transparent**: Shows both raw traces and summaries for full transparency
    
    ### 🔬 About the Multi-Agent System
    
    Our Clinical Decision Support System (CDSS) uses multiple specialized AI agents:
    - **Orchestrator**: Coordinates the workflow and synthesizes findings
    - **Specialist Agents**: Domain experts (cardiology, neurology, infectious disease, etc.)
    - **EXAIM**: Monitors all agent activity and generates compressed summaries
    
    ---
    *Built with ❤️ for safer, more explainable AI in healthcare*
    """)
    
    # Event handlers - use streaming for live updates
    submit_btn.click(
        fn=process_case_gradio,
        inputs=[case_input],
        outputs=[raw_output, summary_output, carousel_output],
        show_progress=False  # Disable progress bar
    )
    
    # Example button handlers
    for btn, example_text in example_buttons:
        btn.click(
            fn=lambda x: x,
            inputs=[gr.State(example_text)],
            outputs=[case_input]
        )


# For Hugging Face Spaces, the demo object is automatically served
# For local development, uncomment the lines below:
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
