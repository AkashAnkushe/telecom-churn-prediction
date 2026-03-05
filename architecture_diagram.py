from graphviz import Digraph

dot = Digraph(format="png")
dot.attr(rankdir="TB")
dot.attr(size="8,10")

# ---------- Layers ----------
dot.attr("node", shape="box", style="filled", color="lightblue")

dot.node("User", "👤 User")

dot.attr("node", shape="box", style="filled", color="lightgreen")
dot.node("UI", "Presentation Layer\nStreamlit UI (app.py)")

dot.attr("node", shape="box", style="filled", color="orange")
dot.node("Predict", "Application Layer\nPrediction Logic (predict.py)")

dot.attr("node", shape="box", style="filled", color="yellow")
dot.node("Model", "ML Layer\nLogistic Regression Model")
dot.node("Pipeline", "Preprocessing Pipeline\n(StandardScaler + OneHotEncoder)")

dot.attr("node", shape="box", style="filled", color="pink")
dot.node("Data", "Data Layer\ntelecom_churn.csv")

# ---------- Connections ----------
dot.edge("User", "UI")
dot.edge("UI", "Predict")
dot.edge("Predict", "Model")
dot.edge("Model", "Pipeline")
dot.edge("Pipeline", "Data")

dot.render("executive_architecture_diagram", view=True)