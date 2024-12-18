import streamlit as st
import pandas as pd
import csv


def historical_results_page():
    # Streamlit App Title
    st.title('Model Performance on Historical Data')

    st.write("Here's how we've performed on historical data (not used in our model's training)")


    # Read the CSV file
    df = pd.read_csv("app_historical_df.csv", encoding="utf-8", sep=",", header = 0)  


    

    # Assuming df is your DataFrame
    # For example:
    # df = pd.read_csv('your_file.csv')
    
    # Create filter widgets in Streamlit for each column
    
    # Filter for home_team dropdown
    home_team_options = st.selectbox(
        "Select Home Team", 
        options=["All"] + list(df['home_team'].unique()),  # Add "All" option
        index=0  # Defaults to the "All" option
    )
    
    # Filter for away_team dropdown
    away_team_options = st.selectbox(
        "Select Away Team", 
        options=["All"] + list(df['away_team'].unique()),  # Add "All" option
        index=0  # Defaults to the "All" option
    )
    
    # Filter for week
    week_options = st.selectbox(
        "Select Week", 
        options=["All"] + list(df['week'].unique()),  # Add "All" option
        index=0  # Defaults to the "All" option
    )
    
    # Filter for season
    season_options = st.selectbox(
        "Select Season", 
        options=["All"] + list(df['season'].unique()),  # Add "All" option
        index=0  # Defaults to the "All" option
    )
    
    # Filter for pred_home_win
    pred_home_win_options = st.selectbox(
        "Select Predicted Home Win", 
        options=["All", 1, 0],  # Add "All" option
        index=0  # Defaults to the "All" option
    )
    
    # Filter for actual_home_win
    actual_home_win_options = st.selectbox(
        "Select Actual Home Win", 
        options=["All", 1, 0],  # Add "All" option
        index=0  # Defaults to the "All" option
    )
    
    # Filter for book_home_spread (manual number input with bounds)
    book_home_spread_lower, book_home_spread_upper = st.slider(
        "Select range for Book Home Spread",
        min_value=int(df['book_home_spread'].min()),
        max_value=int(df['book_home_spread'].max()),
        value=(int(df['book_home_spread'].min()), int(df['book_home_spread'].max()))
    )
    
    # Filter for book_home_moneyline_odds (manual number input with bounds)
    book_home_moneyline_odds_lower, book_home_moneyline_odds_upper = st.slider(
        "Select range for Book Home Moneyline Odds",
        min_value=int(df['book_home_moneyline_odds'].min()),
        max_value=int(df['book_home_moneyline_odds'].max()),
        value=(int(df['book_home_moneyline_odds'].min()), int(df['book_home_moneyline_odds'].max()))
    )
    
    # Filter for pred_home_cover
    pred_home_cover_options = st.selectbox(
        "Select Predicted Home Cover", 
        options=["All", 1, 0],  # Add "All" option
        index=0  # Defaults to the "All" option
    )
    
    # Apply filters based on selections
    
    # Apply home_team filter (including "All" option)
    if home_team_options != "All":
        filtered_df = df[df['home_team'] == home_team_options]
    else:
        filtered_df = df
    
    # Apply away_team filter (including "All" option)
    if away_team_options != "All":
        filtered_df = filtered_df[filtered_df['away_team'] == away_team_options]
        
    # Apply week filter (including "All" option)
    if week_options != "All":
        filtered_df = filtered_df[filtered_df['week'] == week_options]
    
    # Apply season filter (including "All" option)
    if season_options != "All":
        filtered_df = filtered_df[filtered_df['season'] == season_options]
    
    # Apply pred_home_win filter (including "All" option)
    if pred_home_win_options != "All":
        filtered_df = filtered_df[filtered_df['pred_home_win'] == pred_home_win_options]
    
    # Apply actual_home_win filter (including "All" option)
    if actual_home_win_options != "All":
        filtered_df = filtered_df[filtered_df['actual_home_win'] == actual_home_win_options]
    
    # Apply book_home_spread range filter
    filtered_df = filtered_df[
        (filtered_df['book_home_spread'] >= book_home_spread_lower) & 
        (filtered_df['book_home_spread'] <= book_home_spread_upper)
    ]
    
    # Apply book_home_moneyline_odds range filter
    filtered_df = filtered_df[
        (filtered_df['book_home_moneyline_odds'] >= book_home_moneyline_odds_lower) & 
        (filtered_df['book_home_moneyline_odds'] <= book_home_moneyline_odds_upper)
    ]
    
    # Apply pred_home_cover filter (including "All" option)
    if pred_home_cover_options != "All":
        filtered_df = filtered_df[filtered_df['pred_home_cover'] == pred_home_cover_options]
    
    # Display the filtered data
    st.write("### Filtered Data:")
    st.dataframe(filtered_df)
    
    # Option to download filtered data
    csv = filtered_df.to_csv(index=False)
    st.download_button(
        label="Download Filtered Data as CSV",
        data=csv,
        file_name="filtered_data.csv",
        mime="text/csv",
    )



if __name__ == "__main__":
    historical_results_page()
    

# python -m streamlit run cfb_app_launch.py
