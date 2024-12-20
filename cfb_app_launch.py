import streamlit as st
import pandas as pd
import numpy as np
import csv
import math
import xgboost


# Read the CSV files
historical_data = pd.read_csv("app_historical_df.csv", encoding="utf-8", sep=",", header=0)




theor_prepped = pd.read_csv("theoretical_prepped.csv", encoding="utf-8", sep=",", header=0)
theor_prepped = theor_prepped.drop(theor_prepped.columns[0], axis=1)

pdiff_model = xgboost.Booster()
pdiff_model.load_model("cfb_pd_model.bin")

sample_data = pd.read_csv("sample_data.csv", encoding="utf-8", sep=",", header=0)

team_info = pd.read_csv("team_info.csv", encoding="utf-8", sep=",", header=0)








st.set_page_config(layout="wide")



def welcome_page():
  
  st.title('Czar College Football')
  
  st.markdown(
          """
          <style>
          .custom-font {
              font-size: 24px;
          }
          </style>
          <div class="custom-font">Welcome! This site is built around the use of an XGBoost model to predict
          point differential for FBS College Football games. The goal is to predict future outcomes for fun
          and try to beat the accuracy of sportsbook spread models.</div>
          """,
          unsafe_allow_html=True
      )
      
  st.write("  ")
  st.write("  ")
      
  st.markdown(
          """
          <style>
          .custom-font {
              font-size: 24px;
          }
          </style>
          <div class="custom-font">Currently, we are using Version 1 of our model. We are in the process of
          improving it by adding more predictors and re-tuning  for v2. It is not advised to use the model for
          betting yet.</div>
          """,
          unsafe_allow_html=True
      )
      
  st.write("  ")
  st.write("  ")
      
  st.markdown(
          """
          <style>
          .custom-font {
              font-size: 24px;
          }
          </style>
          <div class="custom-font">Check out the Navigation menu on the left to look at how our
          model predicts games (real and theoretical) or how it has performed historically.</div>
          """,
          unsafe_allow_html=True
      )




def historical_results_page():
    # This will make the Streamlit app layout take up the entire width of the browser
    
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
                    In other words, if we evenly split \$100 between the games, we would finish with <span style="color:green"><b>\${our_return_dollars_spread}</b></span>.
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    Using our model to bet the spread in these games would give us a <span style="color:red"><b>{our_return_percentage_spread}</b></span> return on our investment.
                    In other words, if we evenly split \$100 between the games, we would finish with <span style="color:red"><b>\${our_return_dollars_spread}</b></span>.
                """, unsafe_allow_html=True)
            
            
            
            if our_return_spread > naive_return_spread:
                # Create the sentence with the calculated value for blank1
                st.markdown(f"""
                    The average bettor would earn a **{naive_return_percentage_spread}** return on our investment, assuming they win 50% of their bets.
                    If they evenly split \$100 between the games, we would finish with <span><b>\${naive_return_dollars_spread}</b></span>, which is <span style="color:green"><b>${ours_over_naive_spread}</b></span> less than we made.
                """, unsafe_allow_html=True)
            else:
                # Create the sentence with the calculated value for blank1
                st.markdown(f"""
                    The average bettor would earn a **{naive_return_percentage_spread}** return on their investment, assuming they win 50% of their bets.
                    If they evenly split \$100 between the games, they would finish with <span><b>\${naive_return_dollars_spread}</b></span>, which is <span style="color:red"><b>${naive_over_ours_spread}</b></span> more than we made.
                """, unsafe_allow_html=True)
                
            filtered_row_total_spread = filtered_df['spread_winnings'].count()
            
            st.markdown(f"""
                <span><i>using a sample of {filtered_row_total_spread} games for spread calculations</span>
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
                    In other words, if we evenly split \$100 between the games, we would finish with <span style="color:green"><b>\${our_return_dollars_moneyline}</b></span>.
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    Using our model to bet the moneyline in these games would give us a <span style="color:red"><b>{our_return_percentage_moneyline}</b></span> return on our investment.
                    In other words, if we evenly split \$100 between the games, we would finish with <span style="color:red"><b>\${our_return_dollars_moneyline}</b></span>.
                """, unsafe_allow_html=True)
            
            
            
            if our_return_moneyline > naive_return_moneyline:
                # Create the sentence with the calculated value for blank1
                st.markdown(f"""
                    A conservative bettor would earn a **{naive_return_percentage_moneyline}** return on our investment, assuming they always bet the favorite.
                    If they evenly split \$100 between the games, we would finish with <span><b>\${naive_return_dollars_moneyline}</b></span>, which is <span style="color:green"><b>${ours_over_naive_moneyline}</b></span> less than we made.
                """, unsafe_allow_html=True)
            else:
                # Create the sentence with the calculated value for blank1
                st.markdown(f"""
                    A conservative bettor would earn a **{naive_return_percentage_moneyline}** return on their investment, assuming they always bet the favorite.
                    If they evenly split \$100 between the games, they would finish with <span><b>\${naive_return_dollars_moneyline}</b></span>, which is <span style="color:red"><b>${naive_over_ours_moneyline}</b></span> more than we made.
                """, unsafe_allow_html=True)
                
            
            filtered_row_total_moneyline = filtered_df['ml_winnings'].count()
            
            st.markdown(f"""
                <span><i>using a sample of {filtered_row_total_moneyline} games for moneyline calculations</span>
                """, unsafe_allow_html=True)
                
                


def game_predictor_page():
  
  
  # Streamlit App
  st.title("Game Predictor")
  
  st.write("Pick a combination of teams and see what the model predicts if they played today!")

  
  # Create columns for layout
  col1, col2 = st.columns([1,2])
    
  
  
  with col1:
      
      
      st.write("### Filters")
      
      # Step 1: Select Away Team
      away_team = st.selectbox("Select Away Team", theor_prepped["t1_team"])
      
      # Step 2: Select Home Team
      home_team = st.selectbox("Select Home Team", theor_prepped["t1_team"])
      
      
      # Dropdown menu for Yes or No
      neutral_site_input = st.selectbox("Neutral Site?", ["No", "Yes"])
      
      # Encode Yes as 1 and No as 0
      neutral_site_ind = 1 if neutral_site_input == "Yes" else 0
      
    
  with col2:  
    
    
      # away_team = "Indiana"
      # home_team = "Notre Dame"
      # neutral_site_ind = 0
    
      # Filter the DataFrame for the selected teams
      away_team_data = theor_prepped[theor_prepped["t1_team"] == away_team]
      home_team_data = theor_prepped[theor_prepped["t1_team"] == home_team]
    
      # Rename columns for the away team to start with "t2_"
      away_team_data = away_team_data.rename(
          columns={col: col.replace("t1_", "t2_") for col in away_team_data.columns if col.startswith("t1_")}
      )
      away_team_data = away_team_data.drop(columns=["neutral_site", "conference_game", "season", "week", "total_points"])
    
      # Combine into one DataFrame
      combined_row = pd.concat([home_team_data.reset_index(drop=True), away_team_data.reset_index(drop=True)], axis=1)
      
      
      # Get the common columns and order according to df2
      common_cols = [col for col in sample_data.columns if col in combined_row.columns]
      
      # Get columns in df1 that are not in df2
      remaining_cols = [col for col in combined_row.columns if col not in sample_data.columns]
      
      # Reorder df1's columns according to df2, and append the remaining columns
      combined_data = combined_row[common_cols + remaining_cols]
      combined_data = combined_data.drop(columns=["season", "week", "total_points", "t2_home", "t2_win"])
      combined_data.at[0, combined_data.columns[0]] = int(neutral_site_ind)
      combined_data.at[0, combined_data.columns[1]] = int(combined_data.apply(lambda row: 1 if row['t1_conference'] == row['t2_conference'] else 0, axis=1))
      combined_data.at[0, combined_data.columns[2]] = 1
      
    
      
      theor_game_prediction_df = xgboost.DMatrix(combined_data.iloc[:, :109])
      theor_game_prediction = pdiff_model.predict(theor_game_prediction_df)
      
      tgp_result_home = round(float(theor_game_prediction[0]),2)
      tgp_result_away = round(float(theor_game_prediction[0]),2) * -1
      

          
      if tgp_result_home >= 0:
          winning_team = home_team
          losing_team = away_team
      else:
          winning_team = away_team
          losing_team = home_team
                        
                        
      
      win_filter = team_info[team_info['school'] == winning_team]
      winning_logo = win_filter.iloc[0]['logo']
      winning_color = win_filter.iloc[0]['color']
      
      lose_filter = team_info[team_info['school'] == losing_team]
      losing_logo = lose_filter.iloc[0]['logo']
      losing_color = lose_filter.iloc[0]['color']

      
      
      st.markdown(f"""
                            <div style="font-size:35px; font-weight:bold; text-align:center;">
                                {away_team} vs {home_team}...
                            </div>
                        """, unsafe_allow_html=True)
      st.write("  ")
      st.write("  ")
    
      #st.image(winning_logo, width = 200)
      st.markdown("<div style='display: flex; justify-content: center;'><img src='" + winning_logo + "' width='200'></div>", unsafe_allow_html=True)
      
      st.write("  ")
      
      
      if tgp_result_home >= 0:
                    st.markdown(f"""
                            <div style="font-size:35px; font-weight:bold; color:{winning_color}; text-align:center;">
                                {home_team} wins by {tgp_result_home}!
                            </div>
                        """, unsafe_allow_html=True)
      else:
                    st.markdown(f"""
                            <div style="font-size:35px; font-weight:bold; color:{winning_color}; text-align:center;">
                                {away_team} wins by {tgp_result_away}!
                            </div>
                        """, unsafe_allow_html=True)
                
       
       
          
      
  

        

    
    
# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ("Home", "Betting Accuracy", "This Week", "The Playoff", "Game Predictor"))




# Display the selected page
if page == "Home":
    welcome_page()
elif page == "Betting Accuracy":
    historical_results_page()
elif page == "Game Predictor":
    game_predictor_page()

    
  

   

# python -m streamlit run cfb_app_launch.py
