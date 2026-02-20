
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output

# Load data
users = pd.read_csv('../data/users_segmented.csv')
predictions = pd.read_csv('../data/users_with_predictions.csv')

app = Dash(__name__)

COLORS = {
    'High Engagement': '#2ecc71',
    'High Spender': '#3498db',
    'Casual': '#f39c12',
    'At-Risk': '#e74c3c'
}

# --- Layout ---
app.layout = html.Div(style={'fontFamily': 'Arial', 'backgroundColor': '#f4f6f9', 'padding': '20px'}, children=[

    html.H1("🎮 Gaming Churn & Engagement Dashboard",
            style={'textAlign': 'center', 'color': '#2c3e50'}),

    # KPI Cards
    html.Div(style={'display': 'flex', 'justifyContent': 'space-around', 'marginBottom': '30px'}, children=[
        html.Div(style={'background': 'white', 'padding': '20px', 'borderRadius': '10px',
                        'textAlign': 'center', 'width': '20%', 'boxShadow': '2px 2px 8px rgba(0,0,0,0.1)'}, children=[
            html.H2(f"{len(users):,}", style={'color': '#3498db', 'margin': '0'}),
            html.P("Total Users", style={'color': '#7f8c8d'})
        ]),
        html.Div(style={'background': 'white', 'padding': '20px', 'borderRadius': '10px',
                        'textAlign': 'center', 'width': '20%', 'boxShadow': '2px 2px 8px rgba(0,0,0,0.1)'}, children=[
            html.H2(f"{users['churned'].mean():.1%}", style={'color': '#e74c3c', 'margin': '0'}),
            html.P("Churn Rate", style={'color': '#7f8c8d'})
        ]),
        html.Div(style={'background': 'white', 'padding': '20px', 'borderRadius': '10px',
                        'textAlign': 'center', 'width': '20%', 'boxShadow': '2px 2px 8px rgba(0,0,0,0.1)'}, children=[
            html.H2(f"${users['total_purchases'].mean():.2f}", style={'color': '#2ecc71', 'margin': '0'}),
            html.P("Avg Purchases", style={'color': '#7f8c8d'})
        ]),
        html.Div(style={'background': 'white', 'padding': '20px', 'borderRadius': '10px',
                        'textAlign': 'center', 'width': '20%', 'boxShadow': '2px 2px 8px rgba(0,0,0,0.1)'}, children=[
            html.H2(f"{users['engagement_score'].mean():.1f}", style={'color': '#f39c12', 'margin': '0'}),
            html.P("Avg Engagement Score", style={'color': '#7f8c8d'})
        ]),
    ]),

    # Segment filter
    html.Div(style={'marginBottom': '20px', 'textAlign': 'center'}, children=[
        html.Label("Filter by Segment:", style={'fontWeight': 'bold'}),
        dcc.Dropdown(
            id='segment-filter',
            options=[{'label': 'All Segments', 'value': 'All'}] +
                    [{'label': s, 'value': s} for s in users['segment'].unique()],
            value='All',
            style={'width': '300px', 'margin': '10px auto'}
        )
    ]),

    # Charts Row 1
    html.Div(style={'display': 'flex', 'gap': '20px', 'marginBottom': '20px'}, children=[
        html.Div(style={'flex': 1, 'background': 'white', 'borderRadius': '10px', 'padding': '15px'}, children=[
            dcc.Graph(id='segment-pie')
        ]),
        html.Div(style={'flex': 1, 'background': 'white', 'borderRadius': '10px', 'padding': '15px'}, children=[
            dcc.Graph(id='churn-by-segment')
        ]),
    ]),

    # Charts Row 2
    html.Div(style={'display': 'flex', 'gap': '20px', 'marginBottom': '20px'}, children=[
        html.Div(style={'flex': 1, 'background': 'white', 'borderRadius': '10px', 'padding': '15px'}, children=[
            dcc.Graph(id='engagement-scatter')
        ]),
        html.Div(style={'flex': 1, 'background': 'white', 'borderRadius': '10px', 'padding': '15px'}, children=[
            dcc.Graph(id='purchases-box')
        ]),
    ]),
])

# --- Callbacks ---
@app.callback(
    [Output('segment-pie', 'figure'),
     Output('churn-by-segment', 'figure'),
     Output('engagement-scatter', 'figure'),
     Output('purchases-box', 'figure')],
    Input('segment-filter', 'value')
)
def update_charts(selected_segment):
    df = users if selected_segment == 'All' else users[users['segment'] == selected_segment]

    # Pie chart
    seg_counts = df['segment'].value_counts().reset_index()
    seg_counts.columns = ['segment', 'count']
    fig1 = px.pie(seg_counts, names='segment', values='count',
                  title='User Segment Distribution',
                  color='segment', color_discrete_map=COLORS)

    # Churn by segment
    churn_data = df.groupby('segment')['churned'].mean().reset_index()
    churn_data.columns = ['segment', 'churn_rate']
    fig2 = px.bar(churn_data, x='segment', y='churn_rate',
                  title='Churn Rate by Segment',
                  color='segment', color_discrete_map=COLORS)
    fig2.update_yaxes(tickformat='.0%')

    # Engagement vs Purchases scatter
    sample = df.sample(min(2000, len(df)), random_state=42)
    fig3 = px.scatter(sample, x='engagement_score', y='total_purchases',
                      color='segment', color_discrete_map=COLORS,
                      title='Engagement vs Purchases',
                      opacity=0.6)

    # Purchases box
    fig4 = px.box(df, x='segment', y='total_purchases',
                  color='segment', color_discrete_map=COLORS,
                  title='Purchase Distribution by Segment')

    return fig1, fig2, fig3, fig4

if __name__ == '__main__':
    app.run(debug=True)