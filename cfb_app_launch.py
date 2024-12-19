import streamlit as st
import pandas as pd
import numpy as np
import csv
import math

# Read the CSV file
historical_data = pd.read_csv("app_historical_df.csv", encoding="utf-8", sep=",", header=0)

def historical_results_page():
  
    st.set_page_config(layout="wide")  # This will make the Streamlit app layout take up the entire width of the browser
    
    # Streamlit App Title
    st.title('Model Performance on Historical Test Data')

    st.write("This is how well our model would perform on previous games over the last few years versus how well a random 50/50 guesser would perform. Only test data (games not included in model training) are included here. Missing values are in columns are skipped over for return calculations.")

    df = historical_data

    # Clean column names to remove leading/trailing spaces
    df.columns = df.columns.str.strip()

    # Format "season" as a string
    df['season'] = df['season'].astype(str)


    # Get unique teams and sort them in ascending order (for both home_team and away_team)
    unique_teams = pd.unique(df[['home_team', 'away_team']].values.ravel('K'))
    unique_teams_sorted = sorted(unique_teams)

    # Get unique weeks (ascending order)
    unique_weeks = sorted(df['week'].unique())

    # Get unique seasons (descending order)
    unique_seasons = sorted(df['season'].unique(), reverse=True)

    # Create columns for layout
    col1, col2, col3, col4 = st.columns([3, 1, 8, 5])  # Left column for filters, middle column for text, right column for data frame

    with col1:
      
        st.write("### Filters")
        
        # Create dropdowns for filtering options on the left column
        team_options = st.selectbox(
            "Team", 
            options=["All"] + unique_teams_sorted,  # Add "All" option and list unique teams only
            index=0  # Default to the "All" option
        )

        week_options = st.multiselect(
            "Week", 
            options=["All"] + unique_weeks,  # Add "All" option and list unique weeks
            default=["All"]  # Default to the "All" option
        )

        season_options = st.multiselect(
            "Season", 
            options=["All"] + unique_seasons,  # Add "All" option and list unique seasons
            default=["All"]  # Default to the "All" option
        )

        pred_home_win_options = st.selectbox(
            "Predicted Home Team Win?", 
            options=["All", "Yes", "No"],  # Add "All", "Yes", "No" options
            index=0  # Defaults to the "All" option
        )

        actual_home_win_options = st.selectbox(
            "Home Team Actually Won?", 
            options=["All", "Yes", "No"],  # Add "All", "Yes", "No" options
            index=0  # Defaults to the "All" option
        )
        
        # Filter for pred_home_cover (manual dropdown with 1 or 0)
        pred_home_cover_options = st.selectbox(
            "Predicted Home to Cover?", 
            options=["All", "Yes", "No"],  # Add "All", "Yes", "No" options
            index=0  # Defaults to the "All" option
        )

        # Filter for book_home_spread (manual number input with bounds)
        book_home_spread_lower, book_home_spread_upper = st.slider(
            "Book Home Spread",
            min_value=int(math.floor(df['book_home_spread'].min())),
            max_value=int(math.ceil(df['book_home_spread'].max())),
            value=(int(math.floor(df['book_home_spread'].min())), int(math.ceil(df['book_home_spread'].max())))
        )

        # Filter to ask user whether they want to exclude NAs
        exclude_na_spread = st.checkbox("Exclude NAs in the spread columns?", value=False)


        # Filter for book_home_ml_odds (manual number input with bounds)
        book_home_ml_odds_lower, book_home_ml_odds_upper = st.slider(
            "Book Home ML Odds",
            min_value=int(math.floor(df['book_home_ml_odds'].min())),
            max_value=int(math.ceil(df['book_home_ml_odds'].max())),
            value=(int(math.floor(df['book_home_ml_odds'].min())), int(math.ceil(df['book_home_ml_odds'].max())))
        )

        
        
        # Filter to ask user whether they want to exclude NAs
        exclude_na_ml = st.checkbox("Exclude NAs in the ml columns?", value=False)
        
        
        
        # Filter for pred vs book (manual number input with bounds)
        pred_vs_book_lower, pred_vs_book_upper = st.slider(
            "Pred vs Book PD Diff",
            min_value=int(math.floor(df['pred_vs_book_spread'].min())),
            max_value=int(math.ceil(df['pred_vs_book_spread'].max())),
            value=(int(math.floor(df['pred_vs_book_spread'].min())), int(math.ceil(df['pred_vs_book_spread'].max())))
        )
        
    with col2:
        st.write("")
        


    with col4:
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
            (filtered_df['book_home_spread'] <= book_home_spread_upper) |
            (filtered_df['book_home_spread'].isna())
        ]

        filtered_df = filtered_df[
            (filtered_df['book_home_ml_odds'] >= book_home_ml_odds_lower) & 
            (filtered_df['book_home_ml_odds'] <= book_home_ml_odds_upper) |
            (filtered_df['book_home_spread'].isna())
        ]
        
        filtered_df = filtered_df[
            (filtered_df['pred_vs_book_spread'] >= pred_vs_book_lower) & 
            (filtered_df['pred_vs_book_spread'] <= pred_vs_book_upper) |
            (filtered_df['pred_vs_book_spread'].isna())
        ]
        
        
        # Remove "naive_ml_winnings" and "naive_spread_winnings" columns for display
        # columns_to_drop = ['naive_ml_winnings', 'naive_spread_winnings']
        # filtered_df = filtered_df.drop(columns=columns_to_drop, errors='ignore')

        if pred_home_cover_options != "All":
            pred_home_cover_value = 1 if pred_home_cover_options == "Yes" else 0
            filtered_df = filtered_df[filtered_df['pred_home_cover'] == pred_home_cover_value]
            
            
        # Exclude rows with NAs if the user selects the option
        if exclude_na_ml:
            filtered_df = filtered_df.dropna(subset=['ml_winnings'])
            
        if exclude_na_spread:
            filtered_df = filtered_df.dropna(subset=['spread_winnings'])

        # Show the filtered data in Streamlit
        st.write("### Test Data")
        st.dataframe(filtered_df)
        
        
    with col3:
            # Middle column for text (empty for now)
            st.write("### Betting Results")
            
            filtered_row_total = len(filtered_df['spread_winnings'])
            
            st.markdown(f"""
                <i>italic</i>using a sample of {filtered_row_total} games<i>i</i>
            """, unsafe_allow_html=True)
            
            
            
            ### SPREAD
            
            # Calculate blank1: average return on investment
            our_return_spread = filtered_df['spread_winnings'].sum(skipna=True) / filtered_df['spread_winnings'].count()

            # Optionally, format it as a percentage
            our_return_percentage_spread = f"{(our_return_spread * 100) - 100:.2f}%"  # If you want to show it as a percentage
            our_return_dollars_spread = f"{(100 * our_return_spread):.2f}"
            
            # Calculate blank1: average return on investment
            naive_return_spread = filtered_df['naive_spread_winnings'].sum(skipna=True) / filtered_df['naive_spread_winnings'].count()

            # Optionally, format it as a percentage
            naive_return_percentage_spread = f"{(naive_return_spread * 100) - 100:.2f}%"  # If you want to show it as a percentage
            naive_return_dollars_spread = f"{(100 * naive_return_spread):.2f}"
            ours_over_naive_spread = f"{100 * our_return_spread - 100 * naive_return_spread:.2f}"
            naive_over_ours_spread = f"{100 * naive_return_spread - 100 * our_return_spread:.2f}"

            
            
            if our_return_spread >= 1:
                st.markdown(f"""
                        <div style="font-size:30px; font-weight:bold; color:green; text-align:center;">
                            Spread: {our_return_percentage_spread}
                        </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                        <div style="font-size:30px; font-weight:bold; color:red; text-align:center;">
                            Spread: {our_return_percentage_spread}
                        </div>
                    """, unsafe_allow_html=True)
                
            st.write("")



            # Create the sentence with the calculated value for blank1
            if our_return_spread >= 1:
                st.markdown(f"""
                    Using our model to bet the spread in these games would give us a <span style="color:green"><b>{our_return_percentage_spread}</b></span> return on our investment.
                    In other words, if we put \$100 on each game, we would finish with <span style="color:green"><b>\${our_return_dollars_spread}</b></span>.
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    Using our model to bet the spread in these games would give us a <span style="color:red"><b>{our_return_percentage_spread}</b></span> return on our investment.
                    In other words, if we put \$100 on each game, we would finish with <span style="color:red"><b>\${our_return_dollars_spread}</b></span>.
                """, unsafe_allow_html=True)
            
            
            
            if our_return_spread > naive_return_spread:
                # Create the sentence with the calculated value for blank1
                st.markdown(f"""
                    The average bettor would earn a **{naive_return_percentage_spread}** return on our investment, assuming they win 50% of their bets.
                    If they put \$100 on each game, we would finish with <span><b>\${naive_return_dollars_spread}</b></span>, which is <span style="color:green"><b>${ours_over_naive_spread}</b></span> less than we made.
                """, unsafe_allow_html=True)
            else:
                # Create the sentence with the calculated value for blank1
                st.markdown(f"""
                    The average bettor would earn a **{naive_return_percentage_spread}** return on their investment, assuming they win 50% of their bets.
                    If they put \$100 on each game, they would finish with <span><b>\${naive_return_dollars_spread}</b></span>, which is <span style="color:red"><b>${naive_over_ours_spread}</b></span> more than we made.
                """, unsafe_allow_html=True)
                
                
            st.write("")
            st.write("")
                
                
                
            
            ### MONEYLINE
            
            # Calculate blank1: average return on investment
            our_return_moneyline = filtered_df['ml_winnings'].sum(skipna=True) / filtered_df['ml_winnings'].count()

            # Optionally, format it as a percentage
            our_return_percentage_moneyline = f"{(our_return_moneyline * 100) - 100:.2f}%"  # If you want to show it as a percentage
            our_return_dollars_moneyline = f"{(100 * our_return_moneyline):.2f}"
            
            # Calculate blank1: average return on investment
            naive_return_moneyline = filtered_df['naive_ml_winnings'].sum(skipna=True) / filtered_df['naive_ml_winnings'].count()

            # Optionally, format it as a percentage
            naive_return_percentage_moneyline = f"{(naive_return_moneyline * 100) - 100:.2f}%"  # If you want to show it as a percentage
            naive_return_dollars_moneyline = f"{(100 * naive_return_moneyline):.2f}"
            ours_over_naive_moneyline = f"{100 * our_return_moneyline - 100 * naive_return_moneyline:.2f}"
            naive_over_ours_moneyline = f"{100 * naive_return_moneyline - 100 * our_return_moneyline:.2f}"

            
            
            if our_return_moneyline >= 1:
                st.markdown(f"""
                        <div style="font-size:30px; font-weight:bold; color:green; text-align:center;">
                            Moneyline: {our_return_percentage_moneyline}
                        </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                        <div style="font-size:30px; font-weight:bold; color:red; text-align:center;">
                            Moneyline: {our_return_percentage_moneyline}
                        </div>
                    """, unsafe_allow_html=True)
                
            st.write("")



            # Create the sentence with the calculated value for blank1
            if our_return_moneyline >= 1:
                st.markdown(f"""
                    Using our model to bet the moneyline in these games would give us a <span style="color:green"><b>{our_return_percentage_moneyline}</b></span> return on our investment.
                    In other words, if we put \$100 on each game, we would finish with <span style="color:green"><b>\${our_return_dollars_moneyline}</b></span>.
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    Using our model to bet the moneyline in these games would give us a <span style="color:red"><b>{our_return_percentage_moneyline}</b></span> return on our investment.
                    In other words, if we put \$100 on each game, we would finish with <span style="color:red"><b>\${our_return_dollars_moneyline}</b></span>.
                """, unsafe_allow_html=True)
            
            
            
            if our_return_moneyline > naive_return_moneyline:
                # Create the sentence with the calculated value for blank1
                st.markdown(f"""
                    The average bettor would earn a **{naive_return_percentage_moneyline}** return on our investment, assuming they win 50% of their bets.
                    If they put \$100 on each game, we would finish with <span><b>\${naive_return_dollars_moneyline}</b></span>, which is <span style="color:green"><b>${ours_over_naive_moneyline}</b></span> less than we made.
                """, unsafe_allow_html=True)
            else:
                # Create the sentence with the calculated value for blank1
                st.markdown(f"""
                    The average bettor would earn a **{naive_return_percentage_moneyline}** return on their investment, assuming they win 50% of their bets.
                    If they put \$100 on each game, they would finish with <span><b>\${naive_return_dollars_moneyline}</b></span>, which is <span style="color:red"><b>${naive_over_ours_moneyline}</b></span> more than we made.
                """, unsafe_allow_html=True)
                
                
            

    
    
    
    

if __name__ == "__main__":
    historical_results_page()



   

# python -m streamlit run cfb_app_launch.py
