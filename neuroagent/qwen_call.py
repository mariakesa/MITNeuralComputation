from qwen_agent import create_neuro_agent

agent = create_neuro_agent()
result = agent.run("Plot all 3D channel combinations from /home/maria/MITNeuralComputation/ScottLindermanBook/lab01_data.pt")
print(result)
