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
    
    # Calculate overall conversion rate for the filtered data
    total_conversations = len(df_filtered)
    total_converted = len(df_filtered[df_filtered['is_converted'] == 'Yes'])
    overall_conversion_rate = (total_converted / total_conversations * 100) if total_conversations > 0 else 0
    
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
    
    # Add overall conversion rate as a dotted horizontal line (right y-axis)
    fig.add_hline(
        y=overall_conversion_rate,
        line=dict(
            color='gray',
            width=2,
            dash='dot'
        ),
        yref='y2',
        annotation_text=f'Overall CR: {overall_conversion_rate:.1f}%',
        annotation_position='bottom right',
        annotation=dict(
            font=dict(color='gray', size=12),
            showarrow=False,
            xanchor='right',
            bgcolor='white',
            bordercolor='gray',
            borderwidth=1
        )
    )
    
    # Calculate dynamic y-axis range for conversion rate
    max_conversion_rate = conversion_df['Conversion_Rate'].max()
    min_conversion_rate = conversion_df['Conversion_Rate'].min()
    
    # Add some padding to the range, ensuring the overall CR line is visible
    if max_conversion_rate > 0:
        padding = (max_conversion_rate - min_conversion_rate) * 0.1
        y2_max = min(100, max(max_conversion_rate + padding, overall_conversion_rate + 5))
        y2_min = max(0, min(min_conversion_rate - padding, overall_conversion_rate - 5))
    else:
        y2_max = overall_conversion_rate + 5
        y2_min = max(0, overall_conversion_rate - 5)
    
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
            title_font=dict(color='blue'),
            tickfont=dict(color='blue')
        ),
        yaxis2=dict(
            title='Conversion Rate (%)',
            side='right',
            overlaying='y',
            range=[y2_min, y2_max],
            title_font=dict(color='red'),
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

def analyze_message_lengths_by_rule(df):
    """
    Analyze character length of bot messages by rule type and conversion.
    """
    message_data = []
    
    for idx, row in df.iterrows():
        conversation_id = row.get('VAChat_SKEY', idx)
        is_converted = row.get('IsConverted', 'Unknown')
        category = row.get('Rule_No_1', 'Unknown')
        if pd.isna(category):
            category = 'Unknown'
        
        # Analyze bot messages (odd positions, excluding position 1 which is welcome)
        for i in range(3, 22, 2):  # Start from 3 (skip welcome), count odd positions only
            rule_col = f'Rule_No_{i}'
            msg_col = f'Message_No_{i}'
            
            # Check if both rule and message exist and are not empty
            if (rule_col in row and pd.notna(row[rule_col]) and row[rule_col].strip() and
                msg_col in row and pd.notna(row[msg_col]) and row[msg_col].strip()):
                
                rule_name = row[rule_col].strip()
                message_text = row[msg_col].strip()
                char_count = len(message_text)
                
                message_data.append({
                    'conversation_id': conversation_id,
                    'category': category,
                    'rule_name': rule_name,
                    'message_text': message_text,
                    'char_count': char_count,
                    'is_converted': is_converted,
                    'position': i
                })
    
    return message_data

def create_rule_length_analysis(message_data, selected_category=None):
    """
    Create binned analysis of message length vs conversion by rule type.
    """
    df_messages = pd.DataFrame(message_data)
    
    # Filter by category if provided
    if selected_category and selected_category != "All Categories":
        df_filtered = df_messages[df_messages['category'] == selected_category]
        category_text = selected_category
    else:
        df_filtered = df_messages
        category_text = "All Categories"
    
    # Analyze each rule type
    rule_analysis = []
    
    for rule_name in df_filtered['rule_name'].unique():
        rule_messages = df_filtered[df_filtered['rule_name'] == rule_name]
        
        if len(rule_messages) < 5:  # Skip rules with very few instances
            continue
        
        # Create quantile-based bins for this rule
        char_counts = rule_messages['char_count']
        q25, q50, q75 = char_counts.quantile([0.25, 0.5, 0.75])
        
        # Define bins based on quartiles
        bins = [0, q25, q50, q75, char_counts.max() + 1]
        labels = ['Short', 'Medium', 'Long', 'Very Long']
        
        # Handle case where quartiles might be the same
        unique_bins = []
        unique_labels = []
        for i, (bin_val, label) in enumerate(zip(bins, labels + [''])):
            if i == 0 or bin_val > unique_bins[-1]:
                unique_bins.append(bin_val)
                if i < len(labels):
                    unique_labels.append(label)
        
        if len(unique_bins) < 2:
            continue
        
        # Create bins
        rule_messages['length_bin'] = pd.cut(rule_messages['char_count'], 
                                           bins=unique_bins, 
                                           labels=unique_labels, 
                                           include_lowest=True)
        
        # Calculate metrics for each bin
        for bin_name in rule_messages['length_bin'].unique():
            if pd.isna(bin_name):
                continue
                
            bin_data = rule_messages[rule_messages['length_bin'] == bin_name]
            total_count = len(bin_data)
            converted_count = len(bin_data[bin_data['is_converted'] == 'Yes'])
            conversion_rate = (converted_count / total_count * 100) if total_count > 0 else 0
            avg_char_count = bin_data['char_count'].mean()
            
            rule_analysis.append({
                'rule_name': rule_name,
                'length_bin': bin_name,
                'avg_char_count': avg_char_count,
                'total_count': total_count,
                'converted_count': converted_count,
                'conversion_rate': conversion_rate,
                'q25': q25,
                'q50': q50,
                'q75': q75
            })
    
    return pd.DataFrame(rule_analysis), category_text

def create_scatter_plot_by_rule(rule_analysis_df, category_text):
    """
    Create scatter plot showing conversion rate vs average character count by rule type.
    """
    if len(rule_analysis_df) == 0:
        return None
    
    fig = px.scatter(
        rule_analysis_df,
        x='avg_char_count',
        y='conversion_rate',
        color='rule_name',
        size='total_count',
        hover_data=['length_bin', 'converted_count', 'total_count'],
        title=f'Message Length vs Conversion Rate by Rule Type - {category_text}',
        labels={
            'avg_char_count': 'Average Character Count',
            'conversion_rate': 'Conversion Rate (%)',
            'rule_name': 'Rule Type',
            'total_count': 'Sample Size'
        }
    )
    
    fig.update_traces(
        hovertemplate='<b>%{fullData.name}</b><br>' +
                      'Avg Characters: %{x:.0f}<br>' +
                      'Conversion Rate: %{y:.1f}%<br>' +
                      'Length Category: %{customdata[0]}<br>' +
                      'Converted: %{customdata[1]}/%{customdata[2]}<br>' +
                      '<extra></extra>'
    )
    
    fig.update_layout(
        height=500,
        xaxis_title='Average Character Count',
        yaxis_title='Conversion Rate (%)',
        showlegend=True
    )
    
    return fig

def create_box_plot_by_rule(message_data, selected_category=None):
    """
    Create box plot showing character count distribution by rule type and conversion status.
    """
    df_messages = pd.DataFrame(message_data)
    
    # Filter by category if provided
    if selected_category and selected_category != "All Categories":
        df_filtered = df_messages[df_messages['category'] == selected_category]
        category_text = selected_category
    else:
        df_filtered = df_messages
        category_text = "All Categories"
    
    # Filter to rules with sufficient data
    rule_counts = df_filtered['rule_name'].value_counts()
    frequent_rules = rule_counts[rule_counts >= 10].index.tolist()
    df_plot = df_filtered[df_filtered['rule_name'].isin(frequent_rules)]
    
    if len(df_plot) == 0:
        return None, category_text
    
    fig = px.box(
        df_plot,
        x='rule_name',
        y='char_count',
        color='is_converted',
        title=f'Message Length Distribution by Rule Type and Conversion - {category_text}',
        labels={
            'rule_name': 'Rule Type',
            'char_count': 'Character Count',
            'is_converted': 'Converted'
        }
    )
    
    # Update hover template to be more user-friendly
    fig.update_traces(
        hovertemplate='<b>%{fullData.name}</b><br>' +
                      'Rule: %{x}<br>' +
                      'Message Length: %{y} characters<br>' +
                      '<extra></extra>'
    )
    
    fig.update_layout(
        height=500,
        xaxis_tickangle=45,
        xaxis_title='Rule Type',
        yaxis_title='Character Count'
    )
    
    return fig, category_text

def create_combined_box_plot(message_data1, message_data2, selected_category=None):
    """
    Create combined box plot showing both datasets side by side.
    """
    # Prepare data from both datasets
    df1 = pd.DataFrame(message_data1)
    df2 = pd.DataFrame(message_data2)
    
    # Add dataset identifier
    df1['dataset'] = 'Dataset 1'
    df2['dataset'] = 'Dataset 2'
    
    # Combine datasets
    df_combined = pd.concat([df1, df2], ignore_index=True)
    
    # Filter by category if provided
    if selected_category and selected_category != "All Categories":
        df_filtered = df_combined[df_combined['category'] == selected_category]
        category_text = selected_category
    else:
        df_filtered = df_combined
        category_text = "All Categories"
    
    # Filter to rules with sufficient data (at least 10 total across both datasets)
    rule_counts = df_filtered['rule_name'].value_counts()
    frequent_rules = rule_counts[rule_counts >= 10].index.tolist()
    df_plot = df_filtered[df_filtered['rule_name'].isin(frequent_rules)]
    
    if len(df_plot) == 0:
        return None, category_text
    
    # Create combined column for grouping
    df_plot['dataset_conversion'] = df_plot['dataset'] + ' - ' + df_plot['is_converted'].astype(str)
    
    fig = px.box(
        df_plot,
        x='rule_name',
        y='char_count',
        color='dataset_conversion',
        title=f'Message Length Distribution Comparison - {category_text}',
        labels={
            'rule_name': 'Rule Type',
            'char_count': 'Character Count',
            'dataset_conversion': 'Dataset & Conversion'
        },
        color_discrete_map={
            'Dataset 1 - No': '#1f77b4',     # Blue (darker)
            'Dataset 1 - Yes': '#aec7e8',    # Blue (lighter)
            'Dataset 2 - No': '#ff7f0e',     # Orange (darker) 
            'Dataset 2 - Yes': '#ffbb78'     # Orange (lighter)
        }
    )
    
    # Update hover template to be more user-friendly
    fig.update_traces(
        hovertemplate='<b>%{fullData.name}</b><br>' +
                      'Rule: %{x}<br>' +
                      'Message Length: %{y} characters<br>' +
                      '<extra></extra>'
    )
    
    fig.update_layout(
        height=600,
        xaxis_tickangle=45,
        xaxis_title='Rule Type',
        yaxis_title='Character Count',
        legend_title='Dataset & Conversion Status'
    )
    
    return fig, category_text

def debug_uploaded_file(file, file_name="File"):
    """
    Debug function to inspect uploaded file before processing
    """
    if file is None:
        st.error(f"{file_name} is None")
        return False
    
    try:
        # Check file size
        file.seek(0, 2)  # Go to end of file
        file_size = file.tell()
        file.seek(0)  # Reset to beginning
        
        st.write(f"**{file_name} Debug Info:**")
        st.write(f"- File name: {file.name}")
        st.write(f"- File size: {file_size} bytes")
        st.write(f"- File type: {file.type}")
        
        if file_size == 0:
            st.error(f"{file_name} is empty (0 bytes)")
            return False
        
        # Check file extension
        file_ext = file.name.lower().split('.')[-1] if '.' in file.name else ''
        supported_formats = ['csv', 'xls', 'xlsx']
        
        if file_ext not in supported_formats:
            st.warning(f"⚠️ File extension '{file_ext}' might not be supported. Supported: {supported_formats}")
        else:
            st.success(f"✅ File format '{file_ext}' is supported")
        
        return True
                
    except Exception as e:
        st.error(f"❌ Error inspecting {file_name}: {str(e)}")
        return False

def load_and_process_data(file, apply_preprocessing, dataset_name):
    """
    Load and process data with error handling for multiple file formats
    """
    try:
        # First validate the file
        if not debug_uploaded_file(file, dataset_name):
            return None
        
        # Determine file type and read accordingly
        file_ext = file.name.lower().split('.')[-1] if '.' in file.name else ''
        
        if file_ext == 'csv':
            # Try to read the CSV
            df = pd.read_csv(file)
        elif file_ext in ['xls', 'xlsx']:
            # Read Excel file
            # Try to read the first sheet, or specify sheet name if needed
            df = pd.read_excel(file, engine='openpyxl' if file_ext == 'xlsx' else 'xlrd')
        else:
            # Fallback: try CSV first, then Excel
            try:
                file.seek(0)  # Reset file pointer
                df = pd.read_csv(file)
                st.info(f"📄 {dataset_name}: Successfully read as CSV")
            except:
                try:
                    file.seek(0)  # Reset file pointer
                    df = pd.read_excel(file)
                    st.info(f"📄 {dataset_name}: Successfully read as Excel")
                except Exception as e:
                    st.error(f"❌ {dataset_name}: Could not read as CSV or Excel: {str(e)}")
                    return None
        
        if len(df) == 0:
            st.error(f"{dataset_name}: File loaded but contains no rows")
            return None
        
        st.success(f"✅ {dataset_name} loaded successfully: {len(df)} rows, {len(df.columns)} columns")
        
        # Show a preview of column names to help with debugging
        st.write(f"**{dataset_name} Columns Preview:**")
        st.write(f"First 10 columns: {list(df.columns[:10])}")
        if len(df.columns) > 10:
            st.write(f"... and {len(df.columns) - 10} more columns")
        
        if apply_preprocessing:
            with st.spinner(f"Preprocessing {dataset_name}..."):
                df_processed = preprocess_rules(df)
                st.success(f"✅ {dataset_name} preprocessing complete")
        else:
            df_processed = df
        
        return df_processed
        
    except pd.errors.EmptyDataError:
        st.error(f"❌ {dataset_name}: No columns to parse from file. The file appears to be empty or invalid.")
        return None
    except Exception as e:
        st.error(f"❌ Error loading {dataset_name}: {str(e)}")
        st.write(f"**Troubleshooting tips:**")
        st.write("- Ensure the file is a valid CSV, XLS, or XLSX format")
        st.write("- Check that the file is not corrupted")
        st.write("- Verify the file contains the expected column structure")
        return None

def render_length_analysis(file1, file2, apply_preprocessing):
    """
    Render the conversation length analysis content
    """
    results = {}
    
    if file1:
        st.write("### Processing Dataset 1...")
        df1_processed = load_and_process_data(file1, apply_preprocessing, "Dataset 1")
        if df1_processed is not None:
            conv_data1 = analyze_conversation_length(df1_processed)
            results['dataset1'] = {
                'conv_data': conv_data1,
                'df_processed': df1_processed
            }
    
    if file2:
        st.write("### Processing Dataset 2...")
        df2_processed = load_and_process_data(file2, apply_preprocessing, "Dataset 2")
        if df2_processed is not None:
            conv_data2 = analyze_conversation_length(df2_processed)
            results['dataset2'] = {
                'conv_data': conv_data2,
                'df_processed': df2_processed
            }
    
    # Display results
    if 'dataset1' in results:
        st.subheader("Dataset 1")
        conv_data1 = results['dataset1']['conv_data']
        
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
    
    if 'dataset2' in results:
        st.subheader("Dataset 2")
        conv_data2 = results['dataset2']['conv_data']
        
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
    
    # Comparison section (if both files uploaded and processed)
    if 'dataset1' in results and 'dataset2' in results:
        st.header("🔍 Quick Comparison")
        
        conv_data1 = results['dataset1']['conv_data']
        conv_data2 = results['dataset2']['conv_data']
        
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

def get_length_analysis_results(file1, file2, apply_preprocessing):
    """
    Process the data and return results for length analysis (without displaying)
    """
    results = {}
    
    if file1:
        df1_processed = load_and_process_data(file1, apply_preprocessing, "Dataset 1")
        if df1_processed is not None:
            conv_data1 = analyze_conversation_length(df1_processed)
            results['dataset1'] = {
                'conv_data': conv_data1,
                'df_processed': df1_processed
            }
    
    if file2:
        df2_processed = load_and_process_data(file2, apply_preprocessing, "Dataset 2")
        if df2_processed is not None:
            conv_data2 = analyze_conversation_length(df2_processed)
            results['dataset2'] = {
                'conv_data': conv_data2,
                'df_processed': df2_processed
            }
    
    return results

def get_message_analysis_results(file1, file2, apply_preprocessing):
    """
    Process the data and return results for message analysis (without displaying)
    """
    results = {}
    
    if file1:
        df1_processed = load_and_process_data(file1, apply_preprocessing, "Dataset 1")
        if df1_processed is not None:
            message_data1 = analyze_message_lengths_by_rule(df1_processed)
            results['dataset1'] = {
                'message_data': message_data1,
                'df_processed': df1_processed
            }
    
    if file2:
        df2_processed = load_and_process_data(file2, apply_preprocessing, "Dataset 2")
        if df2_processed is not None:
            message_data2 = analyze_message_lengths_by_rule(df2_processed)
            results['dataset2'] = {
                'message_data': message_data2,
                'df_processed': df2_processed
            }
    
    return results

def display_length_analysis_results(results):
    """
    Display the length analysis results (separated from data processing)
    """
    # Display results for dataset1
    if 'dataset1' in results:
        st.subheader("Dataset 1")
        conv_data1 = results['dataset1']['conv_data']
        
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
    
    # Display results for dataset2
    if 'dataset2' in results:
        st.subheader("Dataset 2")
        conv_data2 = results['dataset2']['conv_data']
        
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
    
    # Comparison section (if both files uploaded and processed)
    if 'dataset1' in results and 'dataset2' in results:
        st.header("🔍 Quick Comparison")
        
        conv_data1 = results['dataset1']['conv_data']
        conv_data2 = results['dataset2']['conv_data']
        
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

def display_message_analysis_results(results):
    """
    Display the message analysis results (separated from data processing)
    """
    # Display results for dataset1
    if 'dataset1' in results:
        st.subheader("Dataset 1 - Message Length Analysis")
        message_data1 = results['dataset1']['message_data']
        
        if len(message_data1) == 0:
            st.warning("No bot messages found in Dataset 1")
        else:
            # Category selector
            categories1 = sorted(set([msg['category'] for msg in message_data1]))
            all_options1 = ["All Categories"] + categories1
            selected_category1 = st.selectbox(
                "Select category to analyze:",
                all_options1,
                index=0,
                key="msg_category_selector1",
                help="Choose a specific category or view aggregate across all categories"
            )
            
            # Display summary stats
            df_msg1 = pd.DataFrame(message_data1)
            if selected_category1 != "All Categories":
                df_msg1_filtered = df_msg1[df_msg1['category'] == selected_category1]
            else:
                df_msg1_filtered = df_msg1
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Messages", len(df_msg1_filtered))
            with col2:
                avg_length = df_msg1_filtered['char_count'].mean()
                st.metric("Avg Message Length", f"{avg_length:.0f} chars")
            with col3:
                unique_rules = df_msg1_filtered['rule_name'].nunique()
                st.metric("Unique Rule Types", unique_rules)
            
            # Rule analysis
            rule_analysis1, category_text1 = create_rule_length_analysis(message_data1, selected_category1)
            
            if len(rule_analysis1) > 0:
                # Scatter plot
                st.write("**📈 Conversion Rate vs Message Length by Rule Type**")
                scatter_fig1 = create_scatter_plot_by_rule(rule_analysis1, category_text1)
                if scatter_fig1:
                    st.plotly_chart(scatter_fig1, use_container_width=True)
                
                # Box plot
                st.write("**📦 Message Length Distribution by Rule Type**")
                box_fig1, _ = create_box_plot_by_rule(message_data1, selected_category1)
                if box_fig1:
                    st.plotly_chart(box_fig1, use_container_width=True)
                
                # Summary table
                st.write("**📊 Rule Performance Summary**")
                summary_table1 = rule_analysis1.groupby('rule_name').agg({
                    'total_count': 'sum',
                    'converted_count': 'sum',
                    'avg_char_count': 'mean'
                }).round(1)
                summary_table1['conversion_rate'] = (summary_table1['converted_count'] / summary_table1['total_count'] * 100).round(1)
                summary_table1 = summary_table1.sort_values('total_count', ascending=False)
                st.dataframe(summary_table1, use_container_width=True)
            else:
                st.warning("Insufficient data for rule analysis in the selected category")
    
    # Display results for dataset2
    if 'dataset2' in results:
        st.subheader("Dataset 2 - Message Length Analysis")
        message_data2 = results['dataset2']['message_data']
        
        if len(message_data2) == 0:
            st.warning("No bot messages found in Dataset 2")
        else:
            # Category selector
            categories2 = sorted(set([msg['category'] for msg in message_data2]))
            all_options2 = ["All Categories"] + categories2
            selected_category2 = st.selectbox(
                "Select category to analyze:",
                all_options2,
                index=0,
                key="msg_category_selector2",
                help="Choose a specific category or view aggregate across all categories"
            )
            
            # Display summary stats
            df_msg2 = pd.DataFrame(message_data2)
            if selected_category2 != "All Categories":
                df_msg2_filtered = df_msg2[df_msg2['category'] == selected_category2]
            else:
                df_msg2_filtered = df_msg2
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Messages", len(df_msg2_filtered))
            with col2:
                avg_length = df_msg2_filtered['char_count'].mean()
                st.metric("Avg Message Length", f"{avg_length:.0f} chars")
            with col3:
                unique_rules = df_msg2_filtered['rule_name'].nunique()
                st.metric("Unique Rule Types", unique_rules)
            
            # Rule analysis
            rule_analysis2, category_text2 = create_rule_length_analysis(message_data2, selected_category2)
            
            if len(rule_analysis2) > 0:
                # Scatter plot
                st.write("**📈 Conversion Rate vs Message Length by Rule Type**")
                scatter_fig2 = create_scatter_plot_by_rule(rule_analysis2, category_text2)
                if scatter_fig2:
                    st.plotly_chart(scatter_fig2, use_container_width=True)
                
                # Box plot
                st.write("**📦 Message Length Distribution by Rule Type**")
                box_fig2, _ = create_box_plot_by_rule(message_data2, selected_category2)
                if box_fig2:
                    st.plotly_chart(box_fig2, use_container_width=True)
                
                # Summary table
                st.write("**📊 Rule Performance Summary**")
                summary_table2 = rule_analysis2.groupby('rule_name').agg({
                    'total_count': 'sum',
                    'converted_count': 'sum',
                    'avg_char_count': 'mean'
                }).round(1)
                summary_table2['conversion_rate'] = (summary_table2['converted_count'] / summary_table2['total_count'] * 100).round(1)
                summary_table2 = summary_table2.sort_values('total_count', ascending=False)
                st.dataframe(summary_table2, use_container_width=True)
            else:
                st.warning("Insufficient data for rule analysis in the selected category")
    
    # Combined comparison section (if both datasets exist)
    if 'dataset1' in results and 'dataset2' in results:
        st.header("🔍 Combined Dataset Comparison")
        
        message_data1 = results['dataset1']['message_data']
        message_data2 = results['dataset2']['message_data']
        
        if len(message_data1) > 0 and len(message_data2) > 0:
            # Get common categories for combined analysis
            categories1 = set([msg['category'] for msg in message_data1])
            categories2 = set([msg['category'] for msg in message_data2])
            common_categories = sorted(list(categories1.intersection(categories2)))
            all_categories = sorted(list(categories1.union(categories2)))
            
            # Category selector for combined view
            combined_options = ["All Categories"] + all_categories
            selected_combined_category = st.selectbox(
                "Select category for comparison:",
                combined_options,
                index=0,
                key="combined_category_selector",
                help="Choose a category to compare across both datasets"
            )
            
            # Combined box plot
            st.write("**📦 Side-by-Side Message Length Distribution**")
            st.write("This chart compares message length distributions between both datasets for each rule type.")
            
            combined_fig, combined_category_text = create_combined_box_plot(
                message_data1, 
                message_data2, 
                selected_combined_category
            )
            
            if combined_fig:
                st.plotly_chart(combined_fig, use_container_width=True)
                
                # Character length comparison by rule type
                st.write("**📊 Character Length Comparison by Rule Type**")
                comparison_table = create_rule_comparison_table(message_data1, message_data2, selected_combined_category)
                if comparison_table is not None and len(comparison_table) > 0:
                    st.dataframe(comparison_table, use_container_width=True)
                    
                    # Summary insights text
                    st.write("**💡 Key Insights:**")
                    insights = generate_comparison_insights(comparison_table)
                    for insight in insights:
                        st.write(f"• {insight}")
                else:
                    st.warning("Insufficient common rule types for detailed comparison")
                
                # Summary insights
                col1, col2 = st.columns(2)
                
                # Dataset 1 stats
                df1 = pd.DataFrame(message_data1)
                if selected_combined_category != "All Categories":
                    df1_filtered = df1[df1['category'] == selected_combined_category]
                else:
                    df1_filtered = df1
                
                # Dataset 2 stats
                df2 = pd.DataFrame(message_data2)
                if selected_combined_category != "All Categories":
                    df2_filtered = df2[df2['category'] == selected_combined_category]
                else:
                    df2_filtered = df2
                
                with col1:
                    avg1 = df1_filtered['char_count'].mean()
                    st.metric("Dataset 1 Avg Length", f"{avg1:.0f} chars")
                
                with col2:
                    avg2 = df2_filtered['char_count'].mean()
                    difference = avg2 - avg1
                    st.metric(
                        "Dataset 2 Avg Length", 
                        f"{avg2:.0f} chars",
                        delta=f"{difference:+.0f} chars"
                    )
                
                # Insight
                if abs(difference) > 5:
                    direction = "longer" if difference > 0 else "shorter"
                    st.info(f"💡 Dataset 2 messages are {abs(difference):.0f} characters {direction} on average")
            else:
                st.warning("Insufficient data for combined comparison in the selected category")
        else:
            st.warning("Need data from both datasets for comparison")

def create_rule_comparison_table(message_data1, message_data2, selected_category=None):
    """
    Create a comparison table showing character length differences by rule type between datasets.
    """
    # Convert to DataFrames
    df1 = pd.DataFrame(message_data1)
    df2 = pd.DataFrame(message_data2)
    
    # Filter by category if provided
    if selected_category and selected_category != "All Categories":
        df1 = df1[df1['category'] == selected_category]
        df2 = df2[df2['category'] == selected_category]
    
    # Calculate stats for each rule type in each dataset
    stats1 = df1.groupby('rule_name')['char_count'].agg(['mean', 'count', 'std']).round(1)
    stats1.columns = ['Dataset1_Avg', 'Dataset1_Count', 'Dataset1_Std']
    
    stats2 = df2.groupby('rule_name')['char_count'].agg(['mean', 'count', 'std']).round(1)
    stats2.columns = ['Dataset2_Avg', 'Dataset2_Count', 'Dataset2_Std']
    
    # Merge the stats
    comparison = pd.merge(stats1, stats2, left_index=True, right_index=True, how='outer')
    
    # Fill NaN values with 0 for missing rules
    comparison = comparison.fillna(0)
    
    # Calculate differences
    comparison['Avg_Difference'] = (comparison['Dataset2_Avg'] - comparison['Dataset1_Avg']).round(1)
    comparison['Percent_Difference'] = ((comparison['Dataset2_Avg'] - comparison['Dataset1_Avg']) / comparison['Dataset1_Avg'] * 100).round(1)
    
    # Handle division by zero
    comparison['Percent_Difference'] = comparison['Percent_Difference'].replace([np.inf, -np.inf], 0)
    
    # Filter to rules that exist in both datasets with sufficient data
    comparison = comparison[
        (comparison['Dataset1_Count'] >= 5) & 
        (comparison['Dataset2_Count'] >= 5)
    ]
    
    if len(comparison) == 0:
        return None
    
    # Reorder columns for better readability
    comparison = comparison[[
        'Dataset1_Avg', 'Dataset1_Count',
        'Dataset2_Avg', 'Dataset2_Count', 
        'Avg_Difference', 'Percent_Difference'
    ]]
    
    # Sort by absolute difference (largest differences first)
    comparison['Abs_Difference'] = abs(comparison['Avg_Difference'])
    comparison = comparison.sort_values('Abs_Difference', ascending=False)
    comparison = comparison.drop('Abs_Difference', axis=1)
    
    # Reset index to show rule names as a column
    comparison = comparison.reset_index()
    comparison.columns = [
        'Rule Type', 
        'Dataset 1 Avg (chars)', 'Dataset 1 Count',
        'Dataset 2 Avg (chars)', 'Dataset 2 Count',
        'Difference (chars)', 'Difference (%)'
    ]
    
    return comparison

def generate_comparison_insights(comparison_table):
    """
    Generate text insights from the comparison table.
    """
    insights = []
    
    if len(comparison_table) == 0:
        return ["No common rule types found for comparison"]
    
    # Find biggest differences
    max_increase = comparison_table.loc[comparison_table['Difference (chars)'].idxmax()]
    max_decrease = comparison_table.loc[comparison_table['Difference (chars)'].idxmin()]
    
    # Biggest increase
    if max_increase['Difference (chars)'] > 10:
        insights.append(f"**{max_increase['Rule Type']}** messages got much longer in Dataset 2 (+{max_increase['Difference (chars)']} chars, {max_increase['Difference (%)']:.1f}%)")
    
    # Biggest decrease
    if max_decrease['Difference (chars)'] < -10:
        insights.append(f"**{max_decrease['Rule Type']}** messages got much shorter in Dataset 2 ({max_decrease['Difference (chars)']} chars, {max_decrease['Difference (%)']:.1f}%)")
    
    # Count of rules that got longer vs shorter
    longer_count = len(comparison_table[comparison_table['Difference (chars)'] > 5])
    shorter_count = len(comparison_table[comparison_table['Difference (chars)'] < -5])
    similar_count = len(comparison_table[abs(comparison_table['Difference (chars)']) <= 5])
    
    if longer_count > shorter_count:
        insights.append(f"Most rule types got longer in Dataset 2 ({longer_count} longer, {shorter_count} shorter, {similar_count} similar)")
    elif shorter_count > longer_count:
        insights.append(f"Most rule types got shorter in Dataset 2 ({shorter_count} shorter, {longer_count} longer, {similar_count} similar)")
    else:
        insights.append(f"Rule types changed in both directions ({longer_count} longer, {shorter_count} shorter, {similar_count} similar)")
    
    # Average change
    avg_change = comparison_table['Difference (chars)'].mean()
    if abs(avg_change) > 2:
        direction = "longer" if avg_change > 0 else "shorter"
        insights.append(f"On average, messages are {abs(avg_change):.1f} characters {direction} in Dataset 2")
    
    # Rules with high variability
    high_var_rules = comparison_table[
        (comparison_table['Dataset 1 Avg (chars)'] > 0) & 
        (abs(comparison_table['Difference (%)']) > 20)
    ]
    
    if len(high_var_rules) > 0:
        rule_names = high_var_rules['Rule Type'].head(3).tolist()
        insights.append(f"Biggest relative changes in: {', '.join(rule_names)}")
    
    return insights if insights else ["No significant patterns found"]

def main():
    st.set_page_config(page_title="Conversation Length Analysis", layout="wide")
    
    st.title("Visibility Tools For Bot Components")
    st.markdown("Compare bot response patterns between datasets")
    
    # Sidebar for file uploads
    with st.sidebar:
        st.header("📁 Upload Data")
        
        # First, let user choose how many datasets
        upload_mode = st.radio(
            "How many datasets?",
            ["Single Dataset", "Compare Two Datasets"],
            help="Choose whether to analyze one dataset or compare two"
        )
        
        if upload_mode == "Single Dataset":
            file1 = st.file_uploader(
                "Upload Dataset (CSV/Excel)", 
                type=['csv', 'xls', 'xlsx'], 
                key="file1",
                help="CSV, XLS, or XLSX file with conversation data"
            )
            file2 = None  # Explicitly set to None
            
        else:  # Compare Two Datasets
            file1 = st.file_uploader(
                "Upload Dataset 1 (CSV/Excel)", 
                type=['csv', 'xls', 'xlsx'], 
                key="file1",
                help="CSV, XLS, or XLSX file with conversation data"
            )
            
            file2 = st.file_uploader(
                "Upload Dataset 2 (CSV/Excel)", 
                type=['csv', 'xls', 'xlsx'], 
                key="file2",
                help="CSV, XLS, or XLSX file with conversation data for comparison"
            )
        
        if file1 or file2:
            st.header("⚙️ Processing Options")
            apply_preprocessing = st.checkbox(
                "Apply rule preprocessing", 
                value=True,
                help="Convert repeated rules to sequential format"
            )
        else:
            apply_preprocessing = True  # Default value when no files uploaded
    
    # Initialize session state for persisting results
    if 'length_analysis_results' not in st.session_state:
        st.session_state.length_analysis_results = None
    if 'message_analysis_results' not in st.session_state:
        st.session_state.message_analysis_results = None
    if 'current_files_signature' not in st.session_state:
        st.session_state.current_files_signature = None
    
    # Create a signature for current files to detect changes
    current_signature = ""
    if file1:
        current_signature += f"file1:{file1.name}:{file1.size}"
    if file2:
        current_signature += f"file2:{file2.name}:{file2.size}"
    current_signature += f"preprocessing:{apply_preprocessing}"
    
    # Clear results if files have changed
    if st.session_state.current_files_signature != current_signature:
        st.session_state.length_analysis_results = None
        st.session_state.message_analysis_results = None
        st.session_state.current_files_signature = current_signature
    
    # Create tabs
    tab1, tab2 = st.tabs(["Gambit Number", "Character Count"])
    
    with tab1:
        st.header("📊 Conversation Length Analysis")
        
        if file1 or file2:
            # Show analysis button and check session state
            analysis_button_clicked = st.button("🔍 Start Length Analysis", type="primary", key="start_length_analysis")
            
            # Run analysis if button clicked
            if analysis_button_clicked:
                with st.spinner("Running length analysis..."):
                    st.session_state.length_analysis_results = get_length_analysis_results(file1, file2, apply_preprocessing)
            
            # Display results if they exist
            if st.session_state.length_analysis_results is not None:
                display_length_analysis_results(st.session_state.length_analysis_results)
            else:
                # Show preview info without processing
                st.info("👆 Click the button above to begin analyzing conversation length patterns")
                
                # Show basic file info without processing
                if file1:
                    st.write(f"**Dataset 1:** {file1.name} ({file1.size:,} bytes)")
                if file2:
                    st.write(f"**Dataset 2:** {file2.name} ({file2.size:,} bytes)")
        else:
            st.info("""
            **Getting Started:**
            
            1. Upload one or two CSV/Excel files using the sidebar
            2. Click 'Start Length Analysis' to view conversation length patterns
            3. Compare average bot response counts between datasets
            
            **What this shows:**
            - Average number of bot responses per conversation (excluding welcome messages)
            - Distribution charts showing how conversation length varies by category
            - Category-wise breakdown of conversation patterns
            
            **Supported File Formats:**
            - CSV (.csv)
            - Excel (.xls, .xlsx)
            """)
    
    with tab2:
        st.header("💬 Message Length by Rule Type")
        
        if file1 or file2:
            # Show analysis button and check session state
            message_analysis_button_clicked = st.button("🔍 Start Message Analysis", type="primary", key="start_message_analysis")
            
            # Run analysis if button clicked
            if message_analysis_button_clicked:
                with st.spinner("Running message analysis..."):
                    st.session_state.message_analysis_results = get_message_analysis_results(file1, file2, apply_preprocessing)
            
            # Display results if they exist
            if st.session_state.message_analysis_results is not None:
                display_message_analysis_results(st.session_state.message_analysis_results)
            else:
                # Show preview info without processing
                st.info("👆 Click the button above to begin analyzing message length patterns")
                
                # Show basic file info without processing
                if file1:
                    st.write(f"**Dataset 1:** {file1.name} ({file1.size:,} bytes)")
                if file2:
                    st.write(f"**Dataset 2:** {file2.name} ({file2.size:,} bytes)")
        else:
            st.info("""
            **Message Length Analysis:**
            
            This tab analyzes the relationship between individual bot message length and conversion success.
            
            **What you'll see:**
            - **Scatter Plot**: Shows how conversion rates vary by message length for each rule type
            - **Box Plot**: Compares message length distributions between converted vs non-converted conversations
            - **Rule Performance**: Summary of how each rule type performs at different lengths
            
            **Insights you can gain:**
            - Which rules work better when longer vs shorter
            - Optimal message length ranges for specific rule types
            - How message verbosity affects conversion by category
            
            **Supported File Formats:**
            - CSV (.csv)
            - Excel (.xls, .xlsx)
            
            Upload your files in the sidebar to get started!
            """)


if __name__ == "__main__":
    main()
