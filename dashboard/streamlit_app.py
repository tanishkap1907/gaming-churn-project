
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Gaming Churn Dashboard", page_icon="🎮", layout="wide")

users = pd.read_csv('data/users_segmented.csv')

st.title("🎮 Cloud Gaming Churn & Engagement Dashboard")

# KPI Cards
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Users", f"{len(users):,}")
col2.metric("Churn Rate", f"{users['churned'].mean():.1%}")
col3.metric("Avg Purchases", f"${users['total_purchases'].mean():.2f}")
col4.metric("Avg Engagement", f"{users['engagement_score'].mean():.1f}")

st.divider()

# Filter
segment = st.selectbox("Filter by Segment", ['All'] + list(users['segment'].unique()))
df = users if segment == 'All' else users[users['segment'] == segment]

COLORS = {'High Engagement':'#2ecc71','High Spender':'#3498db',
          'Casual':'#f39c12','At-Risk':'#e74c3c'}

col1, col2 = st.columns(2)

with col1:
    seg_counts = df['segment'].value_counts().reset_index()
    seg_counts.columns = ['segment','count']
    fig1 = px.pie(seg_counts, names='segment', values='count',
                  title='User Segment Distribution', color='segment',
                  color_discrete_map=COLORS)
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    churn_data = df.groupby('segment')['churned'].mean().reset_index()
    churn_data.columns = ['segment','churn_rate']
    fig2 = px.bar(churn_data, x='segment', y='churn_rate',
                  title='Churn Rate by Segment', color='segment',
                  color_discrete_map=COLORS)
    fig2.update_yaxes(tickformat='.0%')
    st.plotly_chart(fig2, use_container_width=True)

col3, col4 = st.columns(2)

with col3:
    sample = df.sample(min(2000, len(df)), random_state=42)
    fig3 = px.scatter(sample, x='engagement_score', y='total_purchases',
                      color='segment', color_discrete_map=COLORS,
                      title='Engagement vs Purchases', opacity=0.6)
    st.plotly_chart(fig3, use_container_width=True)

with col4:
    fig4 = px.box(df, x='segment', y='total_purchases',
                  color='segment', color_discrete_map=COLORS,
                  title='Purchase Distribution by Segment')
    st.plotly_chart(fig4, use_container_width=True)