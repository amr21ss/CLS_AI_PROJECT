import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set page config
st.set_page_config(
    page_title="Diabetes Comprehensive Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Load and preprocess data
@st.cache_data
def load_data():
    # Replace with your actual data loading code
    try:
        df = pd.read_csv('data/diabetes.csv')
    except FileNotFoundError:
        st.error("Data file not found. Please check the file path.")
        st.stop()
    
    # Convert binary columns to strings if they're not already
    binary_cols = ['Diabetes_binary', 'HighChol', 'DiffWalk', 'NoDocbcCost']
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)  # Convert to strings
    
    return df

df = load_data()

# Title and description
st.title("📊 Diabetes Comprehensive Analysis Dashboard")
st.markdown("""
Explore the relationships between diabetes and various health factors using this interactive dashboard.
""")

# Sidebar filters
st.sidebar.header("🔍 Filter Data")

# Age filter
age_options = sorted(df['Age'].unique())
selected_ages = st.sidebar.multiselect(
    "Select Age Groups:",
    options=age_options,
    default=age_options
)

# Health rating filter
if 'GenHlth' in df.columns:
    health_options = sorted(df['GenHlth'].unique())
    selected_health = st.sidebar.multiselect(
        "Select General Health Ratings:",
        options=health_options,
        default=health_options,
        help="1=Excellent, 5=Poor"
    )

# Apply filters
filtered_df = df.copy()
if selected_ages:
    filtered_df = filtered_df[filtered_df['Age'].isin(selected_ages)]
if 'GenHlth' in df.columns and selected_health:
    filtered_df = filtered_df[filtered_df['GenHlth'].isin(selected_health)]

# Color palette (using string keys)
DIABETES_PALETTE = {'0.0': "#66b3ff", '1.0': "#ff9999"}  # Note the string keys

# Main tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Overview", 
    "👥 Demographics", 
    "💊 Health Factors", 
    "📊 Correlations",
    "🧮 BMI Analysis"
])

with tab1:
    st.header("Dataset Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Diabetes Distribution")
        fig, ax = plt.subplots()
        labels = ['Non-Diabetic', 'Diabetic']
        df['Diabetes_binary'].value_counts().plot.pie(
            labels=labels, 
            autopct='%1.1f%%',
            shadow=True, 
            startangle=90,
            colors=[DIABETES_PALETTE['0.0'], DIABETES_PALETTE['1.0']],
            ax=ax
        )
        ax.set_ylabel('')
        st.pyplot(fig)
        
    with col2:
        st.subheader("High Cholesterol Distribution")
        fig, ax = plt.subplots()
        df['HighChol'].astype(str).value_counts().sort_index().plot.bar(
            color=['#88c999', '#fc8d62'],
            ax=ax
        )
        ax.set_xlabel('High Cholesterol (0=No, 1=Yes)')
        ax.set_ylabel('Count')
        st.pyplot(fig)
    
    if 'GenHlth' in df.columns:
        st.subheader("Diabetes Cases by General Health")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.countplot(
            x='GenHlth', 
            hue='Diabetes_binary', 
            data=filtered_df,
            palette=DIABETES_PALETTE,
            ax=ax
        )
        ax.set_title('Diabetes Cases by Self-Reported Health (1=Excellent, 5=Poor)')
        ax.set_xlabel('General Health Rating')
        ax.set_ylabel('Count')
        ax.legend(title='Diabetes', labels=['No', 'Yes'])
        st.pyplot(fig)

with tab2:
    st.header("Demographic Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Diabetes Cases by Age Group")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.countplot(
            x='Age', 
            hue='Diabetes_binary', 
            data=filtered_df,
            palette=DIABETES_PALETTE,
            ax=ax
        )
        ax.set_title('Diabetes Cases by Age')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        ax.legend(title='Diabetes', labels=['No', 'Yes'])
        st.pyplot(fig)
        
    with col2:
        if 'Income' in df.columns and 'NoDocbcCost' in df.columns:
            st.subheader("Healthcare Access by Income")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.countplot(
                x='Income', 
                hue='NoDocbcCost', 
                data=filtered_df,
                palette={'0.0': '#a6d854', '1.0': '#fc8d62'},  # String keys
                ax=ax
            )
            ax.set_title('Could Not Afford Doctor by Income Level')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
            ax.legend(title='Could Not Afford Doctor', labels=['No', 'Yes'])
            st.pyplot(fig)
    
    if 'DiffWalk' in df.columns:
        st.subheader("Walking Difficulty by Diabetes Status")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.countplot(
            x='DiffWalk', 
            hue='Diabetes_binary', 
            data=filtered_df,
            palette=DIABETES_PALETTE,
            ax=ax
        )
        ax.set_title('Walking Difficulty by Diabetes Status')
        ax.set_xlabel('Difficulty Walking (0=No, 1=Yes)')
        ax.legend(title='Diabetes', labels=['No', 'Yes'])
        st.pyplot(fig)

with tab3:
    st.header("Health Factors Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if 'PhysHlth' in df.columns:
            st.subheader("Physical Health Days")
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.boxplot(
                x='Diabetes_binary', 
                y='PhysHlth', 
                data=filtered_df,
                palette=DIABETES_PALETTE,
                ax=ax
            )
            ax.set_title('Physical Health Days by Diabetes Status')
            ax.set_xlabel('Diabetes Status (0=No, 1=Yes)')
            ax.set_ylabel('Days of Poor Physical Health')
            st.pyplot(fig)
        
    with col2:
        if 'MentHlth' in df.columns:
            st.subheader("Mental Health Days")
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.boxplot(
                x='Diabetes_binary', 
                y='MentHlth', 
                data=filtered_df,
                palette=DIABETES_PALETTE,
                ax=ax
            )
            ax.set_title('Mental Health Days by Diabetes Status')
            ax.set_xlabel('Diabetes Status (0=No, 1=Yes)')
            ax.set_ylabel('Days of Poor Mental Health')
            st.pyplot(fig)

with tab4:
    st.header("Correlation Analysis")
    
    st.subheader("Complete Correlation Matrix")
    # Convert relevant columns to numeric for correlation
    corr_df = filtered_df.apply(pd.to_numeric, errors='coerce').dropna(axis=1, how='all')
    fig, ax = plt.subplots(figsize=(20, 15))
    mask = np.triu(np.ones_like(corr_df.corr(), dtype=bool))
    sns.heatmap(
        corr_df.corr(), 
        mask=mask, 
        annot=True, 
        cmap="vlag", 
        fmt='.2f', 
        center=0,
        vmin=-1, 
        vmax=1,
        ax=ax
    )
    ax.set_title('Feature Correlation Matrix', fontsize=16)
    st.pyplot(fig)
    
    st.subheader("Diabetes-Specific Correlations")
    if 'Diabetes_binary' in corr_df.columns:
        diabetes_corr = corr_df.corr()['Diabetes_binary'].sort_values(ascending=False)
        st.dataframe(diabetes_corr, width=600)

with tab5:
    st.header("BMI Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if 'BMI' in df.columns:
            st.subheader("BMI Distribution")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(
                filtered_df['BMI'].dropna(),
                bins=50,
                kde=True,
                color='#8da0cb',
                ax=ax
            )
            ax.set_xlabel('BMI')
            ax.set_ylabel('Count')
            st.pyplot(fig)
            
    with col2:
        if 'BMI' in df.columns:
            st.subheader("BMI by Diabetes Status")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.boxplot(
                x='Diabetes_binary', 
                y='BMI', 
                data=filtered_df,
                palette=DIABETES_PALETTE,
                ax=ax
            )
            ax.set_title('BMI Distribution by Diabetes Status')
            ax.set_xlabel('Diabetes Status (0=No, 1=Yes)')
            ax.set_ylabel('BMI')
            st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown("""
**Note:** This dashboard analyzes the combined diabetes dataset with focus on health indicators.
- Diabetes_binary: 0 = No diabetes, 1 = Prediabetes or diabetes
- HighChol: 0 = No high cholesterol, 1 = High cholesterol
- All binary variables are treated as strings ('0'/'1') for visualization purposes
""")