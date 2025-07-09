import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from collections import defaultdict

def preprocess_rules(df):
    """
    Preprocess rule columns to handle repeated rule names by adding sequential numbers.
    """
    processed_df = df.copy()
    
    # For each conversation (row), track rule occurrences
    for idx, row in processed_df.iterrows():
        rule_counters = defaultdict(int)  # Track count of each rule type
        
        # Process rules in order (1 to 21)
        for i in range(1, 22):
            rule_col = f'Rule_No_{i}'
            if rule_col in row and pd.notna(row[rule_col]) and row[rule_col].strip():
                rule_name = row[rule_col].strip()
                rule_counters[rule_name] += 1
                
                # Create new column name with sequential number
                new_rule_name = f"{rule_name}_{rule_counters[rule_name]}"
                processed_df.at[idx, rule_col] = new_rule_name
    
    return processed_df

def analyze_conversation_length(df):
    """
    Analyze conversation length focusing on bot responses (excluding welcome messages).
    """
    conversation_data = []
    
    for idx, row in df.iterrows():
        # Get category (Rule_No_1 - welcome message)
        category = row.get('Rule_No_1', 'Unknown')
        if pd.isna(category):
            category = 'Unknown'
        
        # Count bot responses (odd positions, excluding position 1 which is welcome)
        bot_response_count = 0
        
        for i in range(3, 22, 2):  # Start from 3 (skip welcome), count odd positions only
            rule_col = f'Rule_No_{i}'
            msg_col = f'Message_No_{i}'
            
            # Check if both rule and message exist and are not empty
            if (rule_col in row and pd.notna(row[rule_col]) and row[rule_col].strip() and
                msg_col in row and pd.notna(row[msg_col]) and row[msg_col].strip()):
                bot_response_count += 1
        
        conversation_data.append({
            'conversation_id': row.get('VAChat_SKEY', idx),
            'category': category,
            'bot_response_count': bot_response_count,
            'is_converted': row.get('IsConverted', 'Unknown')
        })
    
    return conversation_data

def create_conversion_volume_chart(conversation_data, title, selected_category=None):
    """
    Create a dual y-axis line chart showing both volume and conversion rate by response length.
    """
    df_conv = pd.DataFrame(conversation_data)
    
    # Filter by selected category if provided
    if selected_category and selected_category != "All Categories":
        df_filtered = df_conv[df_conv['category'] == selected_category]
        category_text = selected_category
    else:
        df_filtered = df_conv
        category_text = "All Categories"
    
    # Calculate volume and conversion rate by response length
    max_length = df_filtered['bot_response_count'].max()
    
    volume_data = []
    conversion_data = []
    
    for length in range(0, max_length + 1):
        length_conversations = df_filtered[df_filtered['bot_response_count'] == length]
        
        # Volume
        volume = len(length_conversations)
        
        # Conversion rate
        if volume > 0:
            converted = len(length_conversations[length_conversations['is_converted'] == 'Yes'])
            conversion_rate = (converted / volume) * 100
        else:
            conversion_rate = 0
            converted = 0
        
        volume_data.append({
            'Bot_Responses': length,
            'Volume': volume,
            'Converted': converted
        })
        
        conversion_data.append({
            'Bot_Responses': length,
            'Conversion_Rate': conversion_rate,
            'Volume': volume,
            'Converted': converted
        })
    
    volume_df = pd.DataFrame(volume_data)
    conversion_df = pd.DataFrame(conversion_data)
    
    # Create dual y-axis chart
    fig = go.Figure()
    
    # Add volume line (left y-axis)
    fig.add_trace(go.Scatter(
        name='Volume',
        x=volume_df['Bot_Responses'],
        y=volume_df['Volume'],
        mode='lines+markers',
        line=dict(color='blue', width=3),
        marker=dict(size=8, color='blue'),
        yaxis='y',
        hovertemplate='<b>%{x} Bot Responses</b><br>' +
                      'Volume: %{y} conversations<br>' +
                      'Converted: %{customdata}<br>' +
                      '<extra></extra>',
        customdata=volume_df['Converted']
    ))
    
    # Add conversion rate line (right y-axis)
    fig.add_trace(go.Scatter(
        name='Conversion Rate',
        x=conversion_df['Bot_Responses'],
        y=conversion_df['Conversion_Rate'],
        mode='lines+markers',
        line=dict(color='red', width=3),
        marker=dict(size=8, color='red'),
        yaxis='y2',
        hovertemplate='<b>%{x} Bot Responses</b><br>' +
                      'Conversion Rate: %{y:.1f}%<br>' +
                      'Volume: %{customdata[0]} conversations<br>' +
                      'Converted: %{customdata[1]}<br>' +
                      '<extra></extra>',
        customdata=conversion_df[['Volume', 'Converted']].values
    ))
    
    # Calculate dynamic y-axis range for conversion rate
    max_conversion_rate = conversion_df['Conversion_Rate'].max()
    min_conversion_rate = conversion_df['Conversion_Rate'].min()
    
    # Add some padding to the range
    if max_conversion_rate > 0:
        padding = (max_conversion_rate - min_conversion_rate) * 0.1
        y2_max = min(100, max_conversion_rate + padding)
        y2_min = max(0, min_conversion_rate - padding)
    else:
        y2_max = 10
        y2_min = 0
    
    # Update layout
    fig.update_layout(
        title=f"{title} - {category_text}",
        xaxis=dict(
            title='Number of Bot Responses',
            tickmode='linear',
            tick0=0,
            dtick=1
        ),
        yaxis=dict(
            title='Number of Conversations',
            side='left',
            titlefont=dict(color='blue'),
            tickfont=dict(color='blue')
        ),
        yaxis2=dict(
            title='Conversion Rate (%)',
            side='right',
            overlaying='y',
            range=[y2_min, y2_max],
            titlefont=dict(color='red'),
            tickfont=dict(color='red')
        ),
        height=500,
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig, volume_df, conversion_df

def render_length_analysis(file1, file2, apply_preprocessing):
    """
    Render the conversation length analysis content
    """
    if file1:
        st.subheader("Dataset 1")
        
        # Load and process data
        df1 = pd.read_csv(file1)
        
        if apply_preprocessing:
            with st.spinner("Preprocessing Dataset 1..."):
                df1_processed = preprocess_rules(df1)
        else:
            df1_processed = df1
        
        # Analyze conversation length
        conv_data1 = analyze_conversation_length(df1_processed)
        
        # Calculate average bot responses
        avg_bot_responses1 = np.mean([conv['bot_response_count'] for conv in conv_data1])
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Conversations", len(conv_data1))
        with col2:
            st.metric("Avg # of Bot Responses", f"{avg_bot_responses1:.1f}")
        with col3:
            # Category breakdown
            categories1 = [conv['category'] for conv in conv_data1]
            unique_categories1 = len(set(categories1))
            st.metric("Categories", unique_categories1)
        
        # Main chart: Conversion vs Volume by Response Length
        st.write("**📈 Conversion Rate & Volume by Response Length**")
        
        # Category selector
        categories1 = sorted(set([conv['category'] for conv in conv_data1]))
        all_options1 = ["All Categories"] + categories1
        selected_category1 = st.selectbox(
            "Select category to analyze:",
            all_options1,
            index=0,
            key="category_selector1",
            help="Choose a specific category or view aggregate across all categories"
        )
        
        # Create and display chart
        fig1, volume_df1, conversion_df1 = create_conversion_volume_chart(
            conv_data1, 
            "Dataset 1", 
            selected_category1
        )
        st.plotly_chart(fig1, use_container_width=True)
        
        # Summary insights
        if len(conversion_df1) > 0:
            # Find best converting length
            max_conv_row = conversion_df1.loc[conversion_df1['Conversion_Rate'].idxmax()]
            max_volume_row = volume_df1.loc[volume_df1['Volume'].idxmax()]
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "Peak Volume", 
                    f"{int(max_volume_row['Bot_Responses'])} responses",
                    f"{max_volume_row['Volume']} conversations"
                )
            with col2:
                st.metric(
                    "Best Conversion Length", 
                    f"{int(max_conv_row['Bot_Responses'])} responses",
                    f"{max_conv_row['Conversion_Rate']:.1f}%"
                )
            with col3:
                # Calculate total conversion rate for selected category
                total_conv = conversion_df1['Converted'].sum()
                total_vol = conversion_df1['Volume'].sum()
                overall_rate = (total_conv / total_vol * 100) if total_vol > 0 else 0
                st.metric(
                    "Overall Conversion Rate",
                    f"{overall_rate:.1f}%",
                    f"{total_conv}/{total_vol}"
                )
        
        # Category summary table
        st.write("**Category Summary:**")
        df_conv1 = pd.DataFrame(conv_data1)
        category_summary1 = df_conv1.groupby('category').agg({
            'bot_response_count': ['count', 'mean'],
            'conversation_id': 'count'
        }).round(2)
        category_summary1.columns = ['Total Conversations', 'Avg Bot Responses', 'Count Check']
        category_summary1 = category_summary1.drop('Count Check', axis=1)
        st.dataframe(category_summary1, use_container_width=True)
    
    if file2:
        st.divider()
        st.divider()
        st.divider()
        st.subheader("Dataset 2")
        
        # Load and process data
        df2 = pd.read_csv(file2)
        
        if apply_preprocessing:
            with st.spinner("Preprocessing Dataset 2..."):
                df2_processed = preprocess_rules(df2)
        else:
            df2_processed = df2
        
        # Analyze conversation length
        conv_data2 = analyze_conversation_length(df2_processed)
        
        # Calculate average bot responses
        avg_bot_responses2 = np.mean([conv['bot_response_count'] for conv in conv_data2])
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Conversations", len(conv_data2))
        with col2:
            st.metric("Avg # of Bot Responses", f"{avg_bot_responses2:.1f}")
        with col3:
            # Category breakdown
            categories2 = [conv['category'] for conv in conv_data2]
            unique_categories2 = len(set(categories2))
            st.metric("Categories", unique_categories2)
        
        # Main chart: Conversion vs Volume by Response Length
        st.write("**📈 Conversion Rate & Volume by Response Length**")
        
        # Category selector
        categories2 = sorted(set([conv['category'] for conv in conv_data2]))
        all_options2 = ["All Categories"] + categories2
        selected_category2 = st.selectbox(
            "Select category to analyze:",
            all_options2,
            index=0,
            key="category_selector2",
            help="Choose a specific category or view aggregate across all categories"
        )
        
        # Create and display chart
        fig2, volume_df2, conversion_df2 = create_conversion_volume_chart(
            conv_data2, 
            "Dataset 2", 
            selected_category2
        )
        st.plotly_chart(fig2, use_container_width=True)
        
        # Summary insights
        if len(conversion_df2) > 0:
            # Find best converting length
            max_conv_row2 = conversion_df2.loc[conversion_df2['Conversion_Rate'].idxmax()]
            max_volume_row2 = volume_df2.loc[volume_df2['Volume'].idxmax()]
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "Peak Volume", 
                    f"{int(max_volume_row2['Bot_Responses'])} responses",
                    f"{max_volume_row2['Volume']} conversations"
                )
            with col2:
                st.metric(
                    "Best Conversion Length", 
                    f"{int(max_conv_row2['Bot_Responses'])} responses",
                    f"{max_conv_row2['Conversion_Rate']:.1f}%"
                )
            with col3:
                # Calculate total conversion rate for selected category
                total_conv2 = conversion_df2['Converted'].sum()
                total_vol2 = conversion_df2['Volume'].sum()
                overall_rate2 = (total_conv2 / total_vol2 * 100) if total_vol2 > 0 else 0
                st.metric(
                    "Overall Conversion Rate",
                    f"{overall_rate2:.1f}%",
                    f"{total_conv2}/{total_vol2}"
                )
        
        # Category summary table
        st.write("**Category Summary:**")
        df_conv2 = pd.DataFrame(conv_data2)
        category_summary2 = df_conv2.groupby('category').agg({
            'bot_response_count': ['count', 'mean'],
            'conversation_id': 'count'
        }).round(2)
        category_summary2.columns = ['Total Conversations', 'Avg Bot Responses', 'Count Check']
        category_summary2 = category_summary2.drop('Count Check', axis=1)
        st.dataframe(category_summary2, use_container_width=True)
    
    # Comparison section (if both files uploaded)
    if file1 and file2:
        st.header("🔍 Quick Comparison")
        
        # Recalculate averages for comparison
        df1 = pd.read_csv(file1)
        df2 = pd.read_csv(file2)
        
        if apply_preprocessing:
            df1_processed = preprocess_rules(df1)
            df2_processed = preprocess_rules(df2)
        else:
            df1_processed = df1
            df2_processed = df2
        
        conv_data1 = analyze_conversation_length(df1_processed)
        conv_data2 = analyze_conversation_length(df2_processed)
        
        avg_bot_responses1 = np.mean([conv['bot_response_count'] for conv in conv_data1])
        avg_bot_responses2 = np.mean([conv['bot_response_count'] for conv in conv_data2])
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                "Dataset 1 Avg Responses", 
                f"{avg_bot_responses1:.1f}",
                help="Average number of bot responses excluding welcome message"
            )
        
        with col2:
            difference = avg_bot_responses2 - avg_bot_responses1
            st.metric(
                "Dataset 2 Avg Responses", 
                f"{avg_bot_responses2:.1f}",
                delta=f"{difference:+.1f}",
                help="Average number of bot responses excluding welcome message"
            )
        
        # Simple insight
        if abs(difference) > 0.1:
            direction = "longer" if difference > 0 else "shorter"
            st.info(f"💡 Dataset 2 conversations are {abs(difference):.1f} bot responses {direction} on average")

def main():
    st.set_page_config(page_title="Bot Component Analysis", layout="wide")
    
    st.title("👀 Visibility Tools for Pearl")
    st.markdown("Compare bot response behaviors between datasets")
    
    # Sidebar for file uploads
    with st.sidebar:
        st.header("📁 Upload Data")
        
        file1 = st.file_uploader(
            "Upload Dataset 1 (CSV)", 
            type=['csv'], 
            key="file1",
            help="CSV file with conversation data"
        )
        
        file2 = st.file_uploader(
            "Upload Dataset 2 (CSV)", 
            type=['csv'], 
            key="file2",
            help="CSV file with conversation data for comparison"
        )
        
        if file1 or file2:
            st.header("⚙️ Processing Options")
            apply_preprocessing = st.checkbox(
                "Apply rule preprocessing", 
                value=True,
                help="Convert repeated rules to sequential format"
            )
    
    # Create tabs
    tab1, tab2 = st.tabs(["Conversation Length", "Page 2"])
    
    with tab1:
        st.header("📊 Conversation Length Analysis")
        
        if file1 or file2:
            render_length_analysis(file1, file2, apply_preprocessing if (file1 or file2) else True)
        else:
            st.info("""
            **Getting Started:**
            
            1. Upload one or two CSV files using the sidebar
            2. View conversation length patterns by category
            3. Compare average bot response counts between datasets
            
            **What this shows:**
            - Average number of bot responses per conversation (excluding welcome messages)
            - Distribution charts showing how conversation length varies by category
            - Category-wise breakdown of conversation patterns
            """)
    
    with tab2:
        st.header("🔄 Page 2 - Coming Soon")
        st.info("This tab is ready for your next analysis feature!")
        
        # Placeholder content for demonstration
        if file1 or file2:
            st.write("Files uploaded - ready to add new analysis here")
        else:
            st.write("Upload files in the sidebar to enable analysis")

if __name__ == "__main__":
    main()
