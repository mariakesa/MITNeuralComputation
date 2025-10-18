"""
Local neuro-data agent (safe + deterministic, no LLM)
-----------------------------------------------------
• Works with any smolagents version
• Implements its own LocalModel stub
• Calls the plot_all_channels() tool directly
"""

import re
from smolagents import ToolCallingAgent, ChatMessage
from smolagents.models import ApiModel  # base class for compatibility
from tools.plotter import plot_all_channels


# ---------------------------------------------------------------------
#  Minimal local stub model (no GPU, no inference)
# ---------------------------------------------------------------------
class LocalModel(ApiModel):
    """Tiny stand-in for an LLM — echoes requests locally."""

    def __init__(self):
        super().__init__(model_id="local-stub", token=None)

    def create_client(self):
        return None

    def generate(self, prompt, **kwargs):
        # Always return a ChatMessage to match smolagents expectations
        text = f"[LocalModel simulated response] → {prompt[:80]}..."
        return ChatMessage(role="assistant", content=text)


# ---------------------------------------------------------------------
#  Agent definition
# ---------------------------------------------------------------------
class LocalToolAgent(ToolCallingAgent):
    """Local agent using a dummy model and one plotting tool."""

    def __init__(self):
        model = LocalModel()
        super().__init__(
            model=model,
            tools=[plot_all_channels],
            description="Local agent to plot 3D neural channel combinations.",
            add_base_tools=False,
        )

    def run(self, prompt: str):
        """Direct deterministic tool dispatch (no LLM reasoning)."""
        # extract file path like lab01_data.pt or lab01_data.npy
        m = re.search(r'([\w.\-/\\]+?\.(?:pt|npy))', prompt)
        data_path = m.group(1) if m else "lab01_data.pt"

        plot_all_channels(data_path)
        return f"✅ Plotted all 3D channel combinations from {data_path}"


# ---------------------------------------------------------------------
#  Factory
# ---------------------------------------------------------------------
def create_neuro_agent():
    return LocalToolAgent()


if __name__ == "__main__":
    agent = create_neuro_agent()
    print(agent.run("Plot all 3D channel combinations from /home/maria/MITNeuralComputation/ScottLindermanBook/lab01_data.pt"))