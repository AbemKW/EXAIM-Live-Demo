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
        print(f"   Status: {summary.status_action[:50]}...")
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
                output.append("> ")
                output.append("\n> ".join(lines))
                output.append("\n\n")
        
        return "".join(output)
    
    def format_summaries(self) -> str:
        """Format the LATEST summary in a grid-like card structure with dark theme"""
        if not self.summaries:
            return "⏳ **Waiting for summaries...**\n\n*EXAIM will generate summaries as agents complete their reasoning.*"
        
        # Show only the latest summary in grid format
        summary = self.summaries[-1]
        summary_num = len(self.summaries)
        
        output = []
        
        # Header with indicator (dark theme)
        output.append(f"<div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 12px; border-radius: 8px 8px 0 0; margin-bottom: 0;'>\n")
        output.append(f"<h2 style='margin: 0; color: white; font-size: 18px;'>🎯 Latest Summary (#{summary_num})</h2>\n")
        output.append(f"</div>\n\n")
        
        # Grid container for summary fields (dark background)
        output.append("<div style='display: grid; grid-template-columns: 1fr 1fr; gap: 16px; padding: 16px; background: rgba(17, 24, 39, 0.8); border-radius: 0 0 8px 8px;'>\n\n")
        
        # Status/Action - Full width (dark card)
        if summary.status_action:
            output.append("<div style='grid-column: 1 / -1; background: rgba(31, 41, 55, 0.95); padding: 12px; border-radius: 6px; border-left: 4px solid #10b981; color: #f3f4f6;'>\n")
            output.append(f"<strong style='color: #10b981;'>📋 Status/Action</strong>\n\n{summary.status_action}\n")
            output.append("</div>\n\n")
        
        # Key Findings - Left column (dark card)
        if summary.key_findings:
            output.append("<div style='background: rgba(31, 41, 55, 0.95); padding: 12px; border-radius: 6px; border-left: 4px solid #3b82f6; color: #f3f4f6;'>\n")
            output.append(f"<strong style='color: #3b82f6;'>🔍 Key Findings</strong>\n\n{summary.key_findings}\n")
            output.append("</div>\n\n")
        
        # Differential & Rationale - Right column (dark card)
        if summary.differential_rationale:
            output.append("<div style='background: rgba(31, 41, 55, 0.95); padding: 12px; border-radius: 6px; border-left: 4px solid #8b5cf6; color: #f3f4f6;'>\n")
            output.append(f"<strong style='color: #8b5cf6;'>🧠 Differential & Rationale</strong>\n\n{summary.differential_rationale}\n")
            output.append("</div>\n\n")
        
        # Uncertainty/Confidence - Left column (dark card)
        if summary.uncertainty_confidence:
            output.append("<div style='background: rgba(31, 41, 55, 0.95); padding: 12px; border-radius: 6px; border-left: 4px solid #f59e0b; color: #f3f4f6;'>\n")
            output.append(f"<strong style='color: #f59e0b;'>⚖️ Uncertainty/Confidence</strong>\n\n{summary.uncertainty_confidence}\n")
            output.append("</div>\n\n")
        
        # Recommendation - Right column (dark card)
        if summary.recommendation_next_step:
            output.append("<div style='background: rgba(31, 41, 55, 0.95); padding: 12px; border-radius: 6px; border-left: 4px solid #ef4444; color: #f3f4f6;'>\n")
            output.append(f"<strong style='color: #ef4444;'>💡 Recommendation/Next Step</strong>\n\n{summary.recommendation_next_step}\n")
            output.append("</div>\n\n")
        
        # Agent Contributions - Full width (dark card)
        if summary.agent_contributions:
            output.append("<div style='grid-column: 1 / -1; background: rgba(31, 41, 55, 0.95); padding: 12px; border-radius: 6px; border-left: 4px solid #6366f1; color: #f3f4f6;'>\n")
            output.append(f"<strong style='color: #6366f1;'>👥 Agent Contributions</strong>\n\n{summary.agent_contributions}\n")
            output.append("</div>\n\n")
        
        output.append("</div>\n")
        
        return "".join(output)
    
    def format_summary_carousel(self) -> str:
        """Format summaries for the carousel/timeline view - showing only Action field"""
        if not self.summaries:
            return "⏳ **No summaries yet**\n\nSummaries will appear here as they are generated."
        
        output = [f"# 📊 Summary Timeline ({len(self.summaries)} total)\n\n"]
        
        for idx, summary in enumerate(self.summaries, 1):
            # Only show the status_action field for each summary (compact view)
            output.append(f"### Summary #{idx}\n")
            
            if summary.status_action:
                output.append(f"{summary.status_action}\n\n")
            else:
                output.append("*No action specified*\n\n")
            
            output.append("---\n\n")
        
        return "".join(output)


async def process_case_async(case_text: str, handler: GradioStreamingHandler):
    """Process case asynchronously"""
    try:
        handler.processing = True
        handler.reset()
        
        # Create CDSS instance
        cdss = CDSS()
        
        # Register callbacks
        cdss.exaim.register_trace_callback(handler.trace_callback)
        cdss.exaim.register_summary_callback(handler.summary_callback)
        
        # Process case
        await cdss.process_case(case_text, use_streaming=True)
        
    except Exception as e:
        handler.error = str(e)
    finally:
        handler.processing = False


def process_case_gradio(case_text: str):
    """
    Gradio-compatible wrapper for case processing with live streaming
    
    Args:
        case_text: Patient case description
        
    Yields:
        Tuple of (raw_traces_markdown, latest_summary_markdown, carousel_markdown) for each update
    """
    if not case_text or not case_text.strip():
        yield "❌ Please enter a patient case.", "❌ Please enter a patient case.", "❌ Please enter a patient case."
        return
    
    handler = GradioStreamingHandler()
    
    # Start async processing in background thread
    def run_async():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(process_case_async(case_text, handler))
        finally:
            loop.close()
    
    thread = threading.Thread(target=run_async, daemon=True)
    thread.start()
    
    # Yield updates while processing
    last_raw = ""
    last_summary = ""
    last_carousel = ""
    
    # Poll for updates
    while handler.processing or thread.is_alive():
        time.sleep(0.2)  # Check every 200ms
        
        # Check for errors
        if handler.error:
            error_msg = f"❌ **Error processing case:**\n\n```\n{handler.error}\n```"
            yield error_msg, error_msg, error_msg
            return
        
        # Get current state
        current_raw = handler.format_raw_traces()
        current_summary = handler.format_summaries()
        current_carousel = handler.format_summary_carousel()
        
        # Only yield if something changed
        if current_raw != last_raw or current_summary != last_summary or current_carousel != last_carousel:
            last_raw = current_raw
            last_summary = current_summary
            last_carousel = current_carousel
            yield current_raw, current_summary, current_carousel
    
    # Wait for thread to finish
    thread.join(timeout=2)
    
    # Final update
    yield handler.format_raw_traces(), handler.format_summaries(), handler.format_summary_carousel()


# Example cases
EXAMPLE_CASES = [
    """52-year-old male presents to the emergency department with chest pain that started 2 hours ago while watching TV. 
Pain is described as pressure-like, substernal, radiating to left arm and jaw. Associated with shortness of breath, 
diaphoresis, and nausea. Pain not relieved by rest. 

PMH: Hypertension, type 2 diabetes, hyperlipidemia
Medications: Metformin, Lisinopril, Atorvastatin
Social: 20 pack-year smoking history, quit 5 years ago

Vitals: BP 160/95, HR 102, RR 22, O2 sat 94% on RA, Temp 37.2°C

Physical exam: Diaphoretic, anxious-appearing. Heart: Regular rate, no murmurs. Lungs: Clear bilaterally. 
Abdomen: Soft, non-tender.""",

    """67-year-old female with 3-day history of progressive confusion and lethargy. Family reports she has been 
increasingly forgetful and sleeping more than usual. This morning they found her difficult to arouse.

PMH: Hypothyroidism, osteoporosis
Medications: Levothyroxine 50mcg daily, Calcium with Vitamin D
Recent: Started new medication for insomnia 1 week ago (Zolpidem)

Vitals: BP 145/88, HR 58, RR 10, O2 sat 97% on RA, Temp 36.4°C

Physical exam: Lethargic but arousable to voice. Pupils equal and reactive. No focal neurological deficits. 
Lung exam reveals decreased respiratory effort. No signs of trauma.""",

    """8-year-old boy brought in by parents for persistent fever and joint pain for 5 days. Started with high fever 
(39-40°C) and sore throat. Throat symptoms resolved after 2 days but fever persists. Now complaining of painful, 
swollen knees and ankles making it difficult to walk.

PMH: Previously healthy, up-to-date on vaccinations
Recent illness: Had strep throat 3 weeks ago, completed full course of antibiotics

Vitals: BP 105/65, HR 118, RR 24, O2 sat 99% on RA, Temp 39.1°C

Physical exam: Ill-appearing child. Red, swollen bilateral knee and ankle joints with limited range of motion due to pain. 
Faint pink rash on trunk. Heart: Tachycardic but regular, soft systolic murmur at apex (new per parents)."""
]


# Build Gradio interface with custom CSS
custom_css = """
#raw_output {
    overflow-y: auto !important;
    max-height: 600px !important;
    word-wrap: break-word !important;
    white-space: pre-wrap !important;
    overflow-x: hidden !important;
}
#raw_output * {
    max-width: 100% !important;
    word-wrap: break-word !important;
    white-space: pre-wrap !important;
    overflow-wrap: break-word !important;
}
#raw_output pre {
    white-space: pre-wrap !important;
    word-wrap: break-word !important;
    overflow-x: auto !important;
}
#raw_output code {
    white-space: pre-wrap !important;
    word-wrap: break-word !important;
}
#summary_output {
    overflow-y: auto !important;
    max-height: 600px !important;
    word-wrap: break-word !important;
#carousel_output {
    overflow-y: auto !important;
    overflow-x: hidden !important;
    max-height: 250px !important;
    min-height: 250px !important;
    height: 250px !important;
    word-wrap: break-word !important;
    white-space: pre-wrap !important;
}
#carousel_output .prose,
#carousel_output .markdown-body,
#carousel_output > div {
    max-height: 250px !important;
    overflow-y: auto !important;
}
/* Auto-scroll behavior */
.markdown-content {
    scroll-behavior: smooth !important;
}
/* Custom scrollbar styling for all panels */
#raw_output::-webkit-scrollbar,
#summary_output::-webkit-scrollbar,
#carousel_output::-webkit-scrollbar {
    width: 12px;
}
#raw_output::-webkit-scrollbar-track,
#summary_output::-webkit-scrollbar-track,
#carousel_output::-webkit-scrollbar-track {
    background: #e3f2fd;
    border-radius: 10px;
}
#raw_output::-webkit-scrollbar-thumb,
#summary_output::-webkit-scrollbar-thumb,
#carousel_output::-webkit-scrollbar-thumb {
    background: linear-gradient(180deg, #5e6ad2 0%, #4a90e2 100%);
    border-radius: 10px;
    border: 2px solid #e3f2fd;
}
#raw_output::-webkit-scrollbar-thumb:hover,
#summary_output::-webkit-scrollbar-thumb:hover,
#carousel_output::-webkit-scrollbar-thumb:hover {
    background: linear-gradient(180deg, #4a5cc5 0%, #3a7bc8 100%);
}
/* Firefox scrollbar styling */
#raw_output,
#summary_output,
#carousel_output {
    scrollbar-width: thin;
    scrollbar-color: #5e6ad2 #e3f2fd;
}
"""

with gr.Blocks(title="EXAIM - Clinical Decision Support Demo") as demo:
    
    gr.Markdown("""
    # 🏥 EXAIM: Explainable AI Medical Decision Support
    
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


# Configure the demo for both local and Spaces deployment
demo.launch(
    server_name="0.0.0.0",
    server_port=7860,
    share=False,  # Set to True for temporary public link
    theme=gr.themes.Soft(primary_hue="blue", secondary_hue="indigo"),
    css=custom_css,
    # Add custom JavaScript for auto-scrolling all panels
    js="""
    function autoScroll() {
        const rawOutput = document.getElementById('raw_output');
        const summaryOutput = document.getElementById('summary_output');
        const carouselOutput = document.getElementById('carousel_output');
        
        if (rawOutput) {
            rawOutput.scrollTop = rawOutput.scrollHeight;
        }
        if (summaryOutput) {
            summaryOutput.scrollTop = summaryOutput.scrollHeight;
        }
        if (carouselOutput) {
            carouselOutput.scrollTop = carouselOutput.scrollHeight;
        }
    }
    // Call autoScroll periodically to keep all panels scrolled to bottom
    setInterval(autoScroll, 500);
    """
)
