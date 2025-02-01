import networkx as nx
import plotly.graph_objects as go
import random

# Create synthetic data
num_supermarkets = 10
num_foodbanks = 10
data = {}

# Generate random data with sparser connections
# Reduce the probability of connection and ensure some foodbanks have fewer connections
for s in range(num_supermarkets):
    for f in range(num_foodbanks):
        supermarket = f'S{s}'
        foodbank = f'F{f}'
        # Reduce connection probability to 0.3 (30% chance of connection)
        if random.random() < 0.3:
            data[(supermarket, foodbank)] = random.randint(0, 100)

# Ensure at least one connection for each foodbank
for f in range(num_foodbanks):
    foodbank = f'F{f}'
    # Check if foodbank has any connections
    connections = sum(1 for key in data.keys() if key[1] == foodbank)
    if connections == 0:
        # If no connections, create at least one random connection
        random_supermarket = f'S{random.randint(0, num_supermarkets-1)}'
        data[(random_supermarket, foodbank)] = random.randint(0, 100)

# Rest of the code remains the same
G = nx.Graph()

# Add nodes
supermarkets = [f'S{i}' for i in range(num_supermarkets)]
foodbanks = [f'F{i}' for i in range(num_foodbanks)]

G.add_nodes_from(supermarkets, node_type='supermarket')
G.add_nodes_from(foodbanks, node_type='foodbank')

# Add edges
for (s, f), value in data.items():
    G.add_edge(s, f, weight=value)

# Create layout
pos = nx.spring_layout(G)

# Calculate foodbank satisfaction levels
foodbank_satisfaction = {}
for f in foodbanks:
    total_value = sum(data.get((s, f), 0) for s in supermarkets)
    max_possible = 100 * num_supermarkets
    foodbank_satisfaction[f] = total_value / max_possible

# Create edge trace
edge_x = []
edge_y = []
for edge in G.edges():
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    edge_x.extend([x0, x1, None])
    edge_y.extend([y0, y1, None])

edge_trace = go.Scatter(
    x=edge_x, y=edge_y,
    line=dict(width=0.5, color='#888'),
    hoverinfo='none',
    mode='lines')

# Create node traces
supermarket_trace = go.Scatter(
    x=[pos[node][0] for node in supermarkets],
    y=[pos[node][1] for node in supermarkets],
    mode='markers',
    hoverinfo='text',
    text=supermarkets,
    marker=dict(
        color='blue',
        size=20,
        line_width=2))

foodbank_colors = ['rgb({},{},0)'.format(int(255*(1-satisfaction)), int(255*satisfaction)) 
                  for satisfaction in foodbank_satisfaction.values()]

foodbank_trace = go.Scatter(
    x=[pos[node][0] for node in foodbanks],
    y=[pos[node][1] for node in foodbanks],
    mode='markers',
    hoverinfo='text',
    text=[f"{fb} (Fullfillment: {foodbank_satisfaction[fb]:.2%})" for fb in foodbanks],
    marker=dict(
        color=foodbank_colors,
        size=20,
        line_width=2))

# Create figure
fig = go.Figure(data=[edge_trace, supermarket_trace, foodbank_trace])

# Update layout
fig.update_layout(
    title='Food Distribution Network',
    showlegend=False,
    hovermode='closest',
    margin=dict(b=20,l=5,r=5,t=40),
    annotations=[ dict(
        text="Food Distribution Network",
        showarrow=False,
        xref="paper", yref="paper",
        x=0.005, y=-0.002 ) ],
    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))

fig.show()