import streamlit as st
import pandas as pd
import numpy as np
from snowflake.snowpark.context import get_active_session

session = get_active_session()

st.set_page_config(layout="wide")

st.markdown("""
<style>
    .block-container {padding-top: 1rem; padding-bottom: 1rem;}
    [data-testid="stMetric"] {background-color: #f8f9fa; padding: 15px; border-radius: 8px;}
    [data-testid="stMetricValue"] {font-size: 1.8rem;}
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=600)
def load_merchant_summary():
    return session.sql("""
        SELECT 
            MERCHANT_ID, MERCHANT_NAME, MERCHANT_CATEGORY, MERCHANT_CITY,
            MERCHANT_COUNTRY, MERCHANT_LATITUDE, MERCHANT_LONGITUDE,
            COUNT(*) AS TOTAL_TRANSACTIONS,
            SUM(TRANSACTION_AMOUNT_USD) AS TOTAL_REVENUE_USD,
            AVG(TRANSACTION_AMOUNT_USD) AS AVG_TRANSACTION_USD,
            STDDEV(TRANSACTION_AMOUNT_USD) AS STDDEV_TRANSACTION,
            MEDIAN(TRANSACTION_AMOUNT_USD) AS MEDIAN_TRANSACTION,
            COUNT(DISTINCT CUSTOMER_ID) AS UNIQUE_CUSTOMERS,
            COUNT(DISTINCT CARD_ID) AS UNIQUE_CARDS,
            ROUND(100.0 * SUM(CASE WHEN TRANSACTION_STATUS = 'approved' THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 2) AS APPROVAL_RATE,
            SUM(REWARDS_EARNED) AS TOTAL_REWARDS_EARNED,
            MIN(TRANSACTION_DATE) AS FIRST_TRANSACTION,
            MAX(TRANSACTION_DATE) AS LAST_TRANSACTION,
            DATEDIFF('day', MIN(TRANSACTION_DATE), MAX(TRANSACTION_DATE)) AS ACTIVE_DAYS,
            COUNT(*) / NULLIF(DATEDIFF('day', MIN(TRANSACTION_DATE), MAX(TRANSACTION_DATE)), 0) AS TXN_PER_DAY
        FROM CREDIT_CARD_POC.RAW_DATA.TRANSACTIONS
        GROUP BY 1,2,3,4,5,6,7
        ORDER BY TOTAL_REVENUE_USD DESC
    """).to_pandas()

@st.cache_data(ttl=600)
def load_category_summary():
    return session.sql("""
        SELECT MERCHANT_CATEGORY, COUNT(DISTINCT MERCHANT_ID) AS MERCHANT_COUNT,
            COUNT(*) AS TOTAL_TRANSACTIONS, SUM(TRANSACTION_AMOUNT_USD) AS TOTAL_REVENUE_USD,
            AVG(TRANSACTION_AMOUNT_USD) AS AVG_TRANSACTION_USD
        FROM CREDIT_CARD_POC.RAW_DATA.TRANSACTIONS
        GROUP BY 1 ORDER BY TOTAL_REVENUE_USD DESC
    """).to_pandas()

@st.cache_data(ttl=600)
def load_monthly_trends():
    return session.sql("""
        SELECT 
            DATE_TRUNC('month', TRANSACTION_DATE) AS MONTH,
            COUNT(*) AS TRANSACTIONS,
            SUM(TRANSACTION_AMOUNT_USD) AS REVENUE,
            COUNT(DISTINCT CUSTOMER_ID) AS CUSTOMERS,
            COUNT(DISTINCT MERCHANT_ID) AS ACTIVE_MERCHANTS,
            ROUND(100.0 * SUM(CASE WHEN TRANSACTION_STATUS = 'approved' THEN 1 ELSE 0 END) / COUNT(*), 2) AS APPROVAL_RATE
        FROM CREDIT_CARD_POC.RAW_DATA.TRANSACTIONS
        GROUP BY 1 ORDER BY 1
    """).to_pandas()

@st.cache_data(ttl=600)
def load_merchant_monthly():
    return session.sql("""
        SELECT MERCHANT_ID, MERCHANT_NAME, DATE_TRUNC('month', TRANSACTION_DATE) AS MONTH,
            COUNT(*) AS TRANSACTIONS, SUM(TRANSACTION_AMOUNT_USD) AS REVENUE
        FROM CREDIT_CARD_POC.RAW_DATA.TRANSACTIONS
        GROUP BY 1,2,3 ORDER BY 1,3
    """).to_pandas()

def calculate_performance_score(df):
    score_df = df.copy()
    metrics = ['TOTAL_REVENUE_USD', 'UNIQUE_CUSTOMERS', 'APPROVAL_RATE', 'TXN_PER_DAY']
    for m in metrics:
        if m in score_df.columns:
            col = score_df[m].fillna(0)
            min_val, max_val = col.min(), col.max()
            score_df[f'{m}_NORM'] = (col - min_val) / (max_val - min_val) if max_val > min_val else 0.5
    norm_cols = [c for c in score_df.columns if c.endswith('_NORM')]
    score_df['PERFORMANCE_SCORE'] = score_df[norm_cols].mean(axis=1) * 100 if norm_cols else 50
    return score_df

def calculate_growth(trends_df):
    if len(trends_df) >= 2:
        curr = trends_df.iloc[-1]['REVENUE']
        prev = trends_df.iloc[-2]['REVENUE']
        return ((curr - prev) / prev * 100) if prev > 0 else 0
    return 0

merchant_df = load_merchant_summary()
category_df = load_category_summary()
monthly_df = load_monthly_trends()
merchant_monthly = load_merchant_monthly()

merchant_df = calculate_performance_score(merchant_df)
merchant_df['ZSCORE'] = (merchant_df['TOTAL_REVENUE_USD'] - merchant_df['TOTAL_REVENUE_USD'].mean()) / merchant_df['TOTAL_REVENUE_USD'].std()
merchant_df['IS_OUTLIER'] = merchant_df['ZSCORE'].abs() > 2

merchant_monthly = merchant_monthly.sort_values(['MERCHANT_ID', 'MONTH'])
merchant_monthly['PREV_REVENUE'] = merchant_monthly.groupby('MERCHANT_ID')['REVENUE'].shift(1)
merchant_monthly['MOM_GROWTH'] = ((merchant_monthly['REVENUE'] - merchant_monthly['PREV_REVENUE']) / merchant_monthly['PREV_REVENUE'] * 100).round(2)

total_revenue = merchant_df['TOTAL_REVENUE_USD'].sum()
total_merchants = merchant_df['MERCHANT_ID'].nunique()
total_customers = merchant_df['UNIQUE_CUSTOMERS'].sum()
avg_approval = merchant_df['APPROVAL_RATE'].mean()
mom_growth = calculate_growth(monthly_df)

header_col1, header_col2 = st.columns([3, 1])
with header_col1:
    st.title("Merchant Performance Intelligence")
    st.caption(f"Executive Dashboard  •  Data through {merchant_df['LAST_TRANSACTION'].max()}")
with header_col2:
    with st.popover(":material/filter_list: Filters"):
        selected_categories = st.multiselect("Category", merchant_df['MERCHANT_CATEGORY'].unique())
        selected_countries = st.multiselect("Country", merchant_df['MERCHANT_COUNTRY'].unique())
        min_score = st.slider("Min Score", 0, 100, 0)

filtered_df = merchant_df.copy()
if selected_categories:
    filtered_df = filtered_df[filtered_df['MERCHANT_CATEGORY'].isin(selected_categories)]
if selected_countries:
    filtered_df = filtered_df[filtered_df['MERCHANT_COUNTRY'].isin(selected_countries)]
filtered_df = filtered_df[filtered_df['PERFORMANCE_SCORE'] >= min_score]

st.markdown("### Key Performance Indicators")
kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
kpi1.metric("Total Revenue", f"${total_revenue/1e6:.1f}M", f"{mom_growth:+.1f}% MoM")
kpi2.metric("Active Merchants", f"{total_merchants:,}", help="Unique merchants with transactions")
kpi3.metric("Customer Reach", f"{total_customers/1e3:.0f}K", help="Unique customers served")
kpi4.metric("Approval Rate", f"{avg_approval:.1f}%", help="Average transaction approval rate")
kpi5.metric("Avg Performance", f"{filtered_df['PERFORMANCE_SCORE'].mean():.0f}/100", help="Composite merchant score")

st.divider()

main_col1, main_col2 = st.columns([2, 1])

with main_col1:
    st.markdown("### Revenue Trend")
    chart_df = monthly_df.copy()
    chart_df['MONTH'] = pd.to_datetime(chart_df['MONTH'])
    st.area_chart(chart_df.set_index('MONTH')['REVENUE'], use_container_width=True, color="#4285F4")
    
    st.markdown("### Category Performance")
    cat_chart = category_df.head(8).copy()
    cat_chart['REVENUE_M'] = cat_chart['TOTAL_REVENUE_USD'] / 1e6
    st.bar_chart(cat_chart.set_index('MERCHANT_CATEGORY')['REVENUE_M'], use_container_width=True, color="#34A853")

with main_col2:
    st.markdown("### Executive Insights")
    
    top_merchant = filtered_df.loc[filtered_df['PERFORMANCE_SCORE'].idxmax()] if len(filtered_df) > 0 else None
    top_category = category_df.iloc[0]['MERCHANT_CATEGORY'] if len(category_df) > 0 else "N/A"
    outlier_count = filtered_df['IS_OUTLIER'].sum()
    low_performers = len(filtered_df[filtered_df['PERFORMANCE_SCORE'] < 25])
    
    with st.container(border=True):
        st.markdown("**:material/trending_up: Top Performer**")
        if top_merchant is not None:
            st.markdown(f"**{top_merchant['MERCHANT_NAME']}**")
            st.caption(f"Score: {top_merchant['PERFORMANCE_SCORE']:.0f} • Revenue: ${top_merchant['TOTAL_REVENUE_USD']:,.0f}")
    
    with st.container(border=True):
        st.markdown("**:material/category: Leading Category**")
        st.markdown(f"**{top_category}**")
        st.caption(f"${category_df.iloc[0]['TOTAL_REVENUE_USD']/1e6:.1f}M in revenue")
    
    with st.container(border=True):
        st.markdown("**:material/warning: Attention Required**")
        st.markdown(f"**{outlier_count}** statistical outliers")
        st.markdown(f"**{low_performers}** low-score merchants")
    
    with st.container(border=True):
        st.markdown("**:material/lightbulb: Recommendation**")
        if low_performers > 5:
            st.caption("Review underperforming merchants for optimization opportunities")
        elif outlier_count > 0:
            st.caption("Investigate high-revenue outliers for best practice insights")
        else:
            st.caption("Portfolio is healthy. Focus on growth initiatives")

st.divider()

tab1, tab2, tab3 = st.tabs([":material/leaderboard: Leaderboard", ":material/analytics: Deep Analysis", ":material/map: Geographic"])

with tab1:
    lead_col1, lead_col2 = st.columns([3, 2])
    
    with lead_col1:
        st.markdown("#### Merchant Rankings")
        rank_df = filtered_df[['MERCHANT_NAME', 'MERCHANT_CATEGORY', 'PERFORMANCE_SCORE', 
            'TOTAL_REVENUE_USD', 'UNIQUE_CUSTOMERS', 'APPROVAL_RATE']].copy()
        rank_df = rank_df.sort_values('PERFORMANCE_SCORE', ascending=False).head(15)
        rank_df.insert(0, 'RANK', range(1, len(rank_df) + 1))
        st.dataframe(rank_df, column_config={
            "RANK": st.column_config.NumberColumn("Rank", width="small"),
            "MERCHANT_NAME": "Merchant",
            "MERCHANT_CATEGORY": "Category",
            "PERFORMANCE_SCORE": st.column_config.ProgressColumn("Score", min_value=0, max_value=100),
            "TOTAL_REVENUE_USD": st.column_config.NumberColumn("Revenue", format="$%.0f"),
            "UNIQUE_CUSTOMERS": st.column_config.NumberColumn("Customers", format="%d"),
            "APPROVAL_RATE": st.column_config.NumberColumn("Approval", format="%.1f%%")
        }, hide_index=True, use_container_width=True)
    
    with lead_col2:
        st.markdown("#### Score Distribution")
        score_bins = pd.cut(filtered_df['PERFORMANCE_SCORE'], bins=[0,25,50,75,100], labels=['Needs Attention', 'Developing', 'Strong', 'Elite'])
        dist_data = score_bins.value_counts().reindex(['Elite', 'Strong', 'Developing', 'Needs Attention'])
        
        for tier, count in dist_data.items():
            pct = count / len(filtered_df) * 100 if len(filtered_df) > 0 else 0
            color = {"Elite": "green", "Strong": "blue", "Developing": "orange", "Needs Attention": "red"}[tier]
            st.markdown(f":{color}[**{tier}**]: {count} merchants ({pct:.1f}%)")
        
        st.markdown("#### Top Movers")
        latest_growth = merchant_monthly.dropna(subset=['MOM_GROWTH']).sort_values('MONTH').groupby('MERCHANT_NAME').last()
        top_movers = latest_growth.nlargest(5, 'MOM_GROWTH')[['MOM_GROWTH', 'REVENUE']]
        for name, row in top_movers.iterrows():
            st.markdown(f":green[▲] **{name}**: +{row['MOM_GROWTH']:.1f}%")

with tab2:
    analysis_col1, analysis_col2 = st.columns(2)
    
    with analysis_col1:
        st.markdown("#### Statistical Summary")
        stats_data = {
            "Metric": ["Revenue Mean", "Revenue Median", "Revenue Std Dev", "Coefficient of Variation", "Skewness"],
            "Value": [
                f"${filtered_df['TOTAL_REVENUE_USD'].mean():,.0f}",
                f"${filtered_df['TOTAL_REVENUE_USD'].median():,.0f}",
                f"${filtered_df['TOTAL_REVENUE_USD'].std():,.0f}",
                f"{(filtered_df['TOTAL_REVENUE_USD'].std() / filtered_df['TOTAL_REVENUE_USD'].mean() * 100):.1f}%",
                f"{filtered_df['TOTAL_REVENUE_USD'].skew():.2f}"
            ],
            "Interpretation": [
                "Average merchant revenue",
                "Typical merchant revenue",
                "Revenue spread",
                "High = diverse portfolio" if (filtered_df['TOTAL_REVENUE_USD'].std() / filtered_df['TOTAL_REVENUE_USD'].mean() * 100) > 100 else "Consistent revenue",
                "Right-skewed (few high earners)" if filtered_df['TOTAL_REVENUE_USD'].skew() > 1 else "Balanced distribution"
            ]
        }
        st.dataframe(pd.DataFrame(stats_data), hide_index=True, use_container_width=True)
        
        st.markdown("#### Anomaly Summary")
        outliers = filtered_df[filtered_df['IS_OUTLIER']]
        if len(outliers) > 0:
            st.warning(f"{len(outliers)} merchants exceed 2σ threshold")
            for _, row in outliers.head(3).iterrows():
                st.markdown(f"• **{row['MERCHANT_NAME']}**: ${row['TOTAL_REVENUE_USD']:,.0f} (Z={row['ZSCORE']:.1f})")
        else:
            st.success("No statistical anomalies detected")
    
    with analysis_col2:
        st.markdown("#### Category Benchmarks")
        cat_bench = filtered_df.groupby('MERCHANT_CATEGORY').agg({
            'PERFORMANCE_SCORE': 'mean',
            'TOTAL_REVENUE_USD': 'sum',
            'MERCHANT_ID': 'count'
        }).round(1)
        cat_bench.columns = ['Avg Score', 'Total Revenue', 'Merchants']
        cat_bench = cat_bench.sort_values('Avg Score', ascending=False)
        st.dataframe(cat_bench, column_config={
            "Avg Score": st.column_config.ProgressColumn("Avg Score", min_value=0, max_value=100),
            "Total Revenue": st.column_config.NumberColumn("Revenue", format="$%.0f")
        }, use_container_width=True)

with tab3:
    geo_col1, geo_col2 = st.columns([2, 1])
    
    with geo_col1:
        st.markdown("#### Merchant Locations")
        map_df = filtered_df[['MERCHANT_LATITUDE', 'MERCHANT_LONGITUDE', 'TOTAL_REVENUE_USD']].dropna()
        map_df = map_df.rename(columns={'MERCHANT_LATITUDE': 'latitude', 'MERCHANT_LONGITUDE': 'longitude'})
        if not map_df.empty:
            st.map(map_df, size='TOTAL_REVENUE_USD')
    
    with geo_col2:
        st.markdown("#### Revenue by Country")
        country_rev = filtered_df.groupby('MERCHANT_COUNTRY')['TOTAL_REVENUE_USD'].sum().sort_values(ascending=False)
        for country, rev in country_rev.head(8).items():
            pct = rev / country_rev.sum() * 100
            st.markdown(f"**{country}**: ${rev/1e6:.1f}M ({pct:.1f}%)")

st.divider()
st.caption("Merchant Performance Intelligence Dashboard • Auto-refreshes every 10 minutes • Powered by Snowflake")