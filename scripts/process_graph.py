from graphviz import Digraph

# Create a directed graph
dot = Digraph(comment="CSP ML + Tail-Risk Workflow", format="png")
dot.attr(rankdir="LR", size="8")

# Nodes
dot.node("raw", "Raw CSP Trade Data (CSV)", shape="box", style="filled", fillcolor="#f9f9f9")
dot.node("feat", "Analyze Features\n(analyze_csp_features.py)", shape="box", style="filled", fillcolor="#f9f9f9")
dot.node("label", "labeled_trades.csv", shape="box", style="filled", fillcolor="#f0f0f0")
dot.node("gex", "Merge GEX Features\n(merge_gex_features.py)", shape="box", style="filled", fillcolor="#f9f9f9")
dot.node("gex_label", "labeled_trades_with_gex.csv", shape="box", style="filled", fillcolor="#f0f0f0")

dot.node("winner", "Winner Model Training\n(train_csp_model_and_sim.py)", shape="box", style="filled", fillcolor="#e0f7fa")
dot.node("winner_out", "scored_trades_full.csv", shape="box", style="filled", fillcolor="#f0f0f0")

dot.node("tail", "Tail-Loss Model Training\n(train_tail_with_gex.py)", shape="box", style="filled", fillcolor="#ffe0b2")
dot.node("tail_score", "Tail Probabilities\n(score_tail_with_gex.py)", shape="box", style="filled", fillcolor="#f0f0f0")
dot.node("tail_label", "labeled_trades_with_gex_tail.csv", shape="box", style="filled", fillcolor="#f0f0f0")

dot.node("filter", "Rule-Based / Hybrid Filter\n(evaluate_csp_filters.py)", shape="box", style="filled", fillcolor="#dcedc8")
dot.node("sim", "Portfolio Simulation\n(train_csp_model_and_sim.py)", shape="box", style="filled", fillcolor="#c8e6c9")

# Edges
dot.edge("raw", "feat")
dot.edge("feat", "label")
dot.edge("label", "winner")
dot.edge("winner", "winner_out")
dot.edge("label", "gex")
dot.edge("gex", "gex_label")
dot.edge("gex_label", "tail")
dot.edge("tail", "tail_score")
dot.edge("tail_score", "tail_label")
dot.edge("tail_label", "filter")
dot.edge("winner_out", "filter")
dot.edge("filter", "sim")

# Render the diagram to file
output_path = './csp_workflow_diagram'
dot.render(output_path, cleanup=True)

output_path + ".png"
