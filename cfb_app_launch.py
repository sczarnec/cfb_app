import streamlit as st
import pandas as pd
import csv

# Read the CSV file
historical_data = pd.read_csv("app_historical_df.csv", encoding="utf-8", sep=",", header=0)

def historical_results_page():
  
    st.set_page_config(layout="wide")  # This will make the Streamlit app layout take up the entire width of the browser
    
    # Streamlit App Title
    st.title('Model Performance on Historical Data')

    st.write("Here's how we've performed on historical data (not used in our model's training)")

    df = historical_data

    # Clean column names to remove leading/trailing spaces
    df.columns = df.columns.str.strip()

    # Format "season" as a string
    df['season'] = df['season'].astype(str)

    # Remove "naive_ml_winnings" and "naive_spread_winnings" columns for display
    columns_to_drop = ['naive_ml_winnings', 'naive_spread_winnings']
    df = df.drop(columns=columns_to_drop, errors='ignore')

    # Get unique teams and sort them in ascending order (for both home_team and away_team)
    unique_teams = pd.unique(df[['home_team', 'away_team']].values.ravel('K'))
    unique_teams_sorted = sorted(unique_teams)

    # Get unique weeks (ascending order)
    unique_weeks = sorted(df['week'].unique())

    # Get unique seasons (descending order)
    unique_seasons = sorted(df['season'].unique(), reverse=True)

    # Create columns for layout
    col1, col2, col3 = st.columns([1, 3, 3])  # Left column for filters, middle column for text, right column for data frame

    with col1:
        # Create dropdowns for filtering options on the left column
        team_options = st.selectbox(
            "Select Team", 
            options=["All"] + unique_teams_sorted,  # Add "All" option and list unique teams only
            index=0  # Default to the "All" option
        )

        week_options = st.multiselect(
            "Select Week", 
            options=["All"] + unique_weeks,  # Add "All" option and list unique weeks
            default=["All"]  # Default to the "All" option
        )

        season_options = st.multiselect(
            "Select Season", 
            options=["All"] + unique_seasons,  # Add "All" option and list unique seasons
            default=["All"]  # Default to the "All" option
        )

        pred_home_win_options = st.selectbox(
            "Select Predicted Home Win", 
            options=["All", "Yes", "No"],  # Add "All", "Yes", "No" options
            index=0  # Defaults to the "All" option
        )

        actual_home_win_options = st.selectbox(
            "Select Actual Home Win", 
            options=["All", "Yes", "No"],  # Add "All", "Yes", "No" options
            index=0  # Defaults to the "All" option
        )

        # Filter for book_home_spread (manual number input with bounds)
        book_home_spread_lower, book_home_spread_upper = st.slider(
            "Select range for Book Home Spread",
            min_value=int(df['book_home_spread'].min()),
            max_value=int(df['book_home_spread'].max()),
            value=(int(df['book_home_spread'].min()), int(df['book_home_spread'].max()))
        )

        # Filter for book_home_ml_odds (manual number input with bounds)
        book_home_ml_odds_lower, book_home_ml_odds_upper = st.slider(
            "Select range for Book Home ml Odds",
            min_value=int(df['book_home_ml_odds'].min()),
            max_value=int(df['book_home_ml_odds'].max()),
            value=(int(df['book_home_ml_odds'].min()), int(df['book_home_ml_odds'].max()))
        )

        # Filter for pred_home_cover (manual dropdown with 1 or 0)
        pred_home_cover_options = st.selectbox(
            "Select Predicted Home Cover", 
            options=["All", "Yes", "No"],  # Add "All", "Yes", "No" options
            index=0  # Defaults to the "All" option
        )

    with col2:
        # Middle column for text (empty for now)
        st.write("### Middle Column (Text Placeholder)")
        st.write("This is where you can add more content or text as needed.")

    with col3:
        # Apply filters based on selections
        filtered_df = df

        # Apply team filter (home_team or away_team)
        if team_options != "All":
            filtered_df = filtered_df[filtered_df['home_team'].eq(team_options) | filtered_df['away_team'].eq(team_options)]
        else:
            filtered_df = df


        if "All" not in week_options:
            filtered_df = filtered_df[filtered_df['week'].isin(week_options)]

        if "All" not in season_options:
            filtered_df = filtered_df[filtered_df['season'].isin(season_options)]

        if pred_home_win_options != "All":
            pred_home_win_value = 1 if pred_home_win_options == "Yes" else 0
            filtered_df = filtered_df[filtered_df['pred_home_win'] == pred_home_win_value]

        if actual_home_win_options != "All":
            actual_home_win_value = 1 if actual_home_win_options == "Yes" else 0
            filtered_df = filtered_df[filtered_df['actual_home_win'] == actual_home_win_value]

        filtered_df = filtered_df[
            (filtered_df['book_home_spread'] >= book_home_spread_lower) & 
            (filtered_df['book_home_spread'] <= book_home_spread_upper)
        ]

        filtered_df = filtered_df[
            (filtered_df['book_home_ml_odds'] >= book_home_ml_odds_lower) & 
            (filtered_df['book_home_ml_odds'] <= book_home_ml_odds_upper)
        ]

        if pred_home_cover_options != "All":
            pred_home_cover_value = 1 if pred_home_cover_options == "Yes" else 0
            filtered_df = filtered_df[filtered_df['pred_home_cover'] == pred_home_cover_value]

        # Show the filtered data in Streamlit
        st.write("### Filtered Data")
        st.dataframe(filtered_df)

if __name__ == "__main__":
    historical_results_page()



   

# python -m streamlit run cfb_app_launch.py
