import streamlit as st
import pandas as pd
import numpy as np
import csv
import math
import xgboost



#### READ IN CSVs

# Read in historical betting data
historical_data = pd.read_csv("app_historical_df.csv", encoding="utf-8", sep=",", header=0)


# load theoretical data
theor_prepped = pd.read_csv("theoretical_prepped.csv", encoding="utf-8", sep=",", header=0)
theor_prepped = theor_prepped.drop(theor_prepped.columns[0], axis=1)

# load point diff model
pdiff_model = xgboost.Booster()
pdiff_model.load_model("cfb_pd_model.bin")

# load sample col order for xgb
sample_data = pd.read_csv("sample_data.csv", encoding="utf-8", sep=",", header=0)

# load team info data
team_info = pd.read_csv("team_info.csv", encoding="utf-8", sep=",", header=0)



def game_predictor_func(home, away, neutral):
  
        # Filter the DataFrame for the selected teams
      away_team_data = theor_prepped[theor_prepped["t1_team"] == away]
      home_team_data = theor_prepped[theor_prepped["t1_team"] == home]
    
      # Rename columns for the away team to start with "t2_"
      away_team_data = away_team_data.rename(
          columns={col: col.replace("t1_", "t2_") for col in away_team_data.columns if col.startswith("t1_")}
      )
      away_team_data = away_team_data.drop(columns=["neutral_site", "conference_game", "season", "week", "total_points"])
    
      # Combine into one row
      combined_row = pd.concat([home_team_data.reset_index(drop=True), away_team_data.reset_index(drop=True)], axis=1)
      
      
      # Get the common columns and order according to df2
      common_cols = [col for col in sample_data.columns if col in combined_row.columns]
      
      # Get columns in df1 that are not in df2
      remaining_cols = [col for col in combined_row.columns if col not in sample_data.columns]
      
      # Reorder df1's columns according to df2, and append the remaining columns
      # add back in neutral site and conference game data, make ture t1_home is 1
      combined_data = combined_row[common_cols + remaining_cols]
      combined_data = combined_data.drop(columns=["season", "week", "total_points", "t2_home", "t2_win"])
      combined_data.at[0, combined_data.columns[0]] = int(neutral)
      combined_data.at[0, combined_data.columns[1]] = int(combined_data.apply(lambda row: 1 if row['t1_conference'] == row['t2_conference'] else 0, axis=1))
      combined_data.at[0, combined_data.columns[2]] = 1
      
    
      # predict new game
      theor_game_prediction_df = xgboost.DMatrix(combined_data.iloc[:, :109])
      theor_game_prediction = pdiff_model.predict(theor_game_prediction_df)
      
      # results for home and away
      tgp_result_home = round(float(theor_game_prediction[0]),2)
      tgp_result_away = round(float(theor_game_prediction[0]),2) * -1
      
      
      
      
        # Filter the DataFrame for the selected teams
      away_team_data_opp = theor_prepped[theor_prepped["t1_team"] == home]
      home_team_data_opp = theor_prepped[theor_prepped["t1_team"] == away]
    
      # Rename columns for the away team to start with "t2_"
      away_team_data_opp = away_team_data_opp.rename(
          columns={col: col.replace("t1_", "t2_") for col in away_team_data_opp.columns if col.startswith("t1_")}
      )
      away_team_data_opp = away_team_data_opp.drop(columns=["neutral_site", "conference_game", "season", "week", "total_points"])
    
      # Combine into one row
      combined_row_opp = pd.concat([home_team_data_opp.reset_index(drop=True), away_team_data_opp.reset_index(drop=True)], axis=1)
      
      
      # Get the common columns and order according to df2
      common_cols_opp = [col for col in sample_data.columns if col in combined_row_opp.columns]
      
      # Get columns in df1 that are not in df2
      remaining_cols_opp = [col for col in combined_row_opp.columns if col not in sample_data.columns]
      
      # Reorder df1's columns according to df2, and append the remaining columns
      # add back in neutral site and conference game data, make ture t1_home is 1
      combined_data_opp = combined_row_opp[common_cols_opp + remaining_cols_opp]
      combined_data_opp = combined_data_opp.drop(columns=["season", "week", "total_points", "t2_home", "t2_win"])
      combined_data_opp.at[0, combined_data_opp.columns[0]] = int(neutral)
      combined_data_opp.at[0, combined_data_opp.columns[1]] = int(combined_data_opp.apply(lambda row: 1 if row['t1_conference'] == row['t2_conference'] else 0, axis=1))
      combined_data_opp.at[0, combined_data_opp.columns[2]] = 1
      
    
      # predict new game
      theor_game_prediction_df_opp = xgboost.DMatrix(combined_data_opp.iloc[:, :109])
      theor_game_prediction_opp = pdiff_model.predict(theor_game_prediction_df_opp)
      
      # results for home and away
      tgp_result_home_opp = round(float(theor_game_prediction_opp[0]),2)
      tgp_result_away_opp = round(float(theor_game_prediction_opp[0]),2) * -1
      
      # home = "Notre Dame"
      # away = "Indiana"
      # neutral = 1
      
      if neutral == 0:
            # define winning/losing teams
            if tgp_result_home >= 0:
                winning_team = home
                losing_team = away
                tgp_result_wt = tgp_result_home
            else:
                winning_team = away
                losing_team = home
                tgp_result_wt = tgp_result_away
                              
                              
            # winning and losing team info
            win_filter = team_info[team_info['school'] == winning_team]
            winning_logo = win_filter.iloc[0]['logo']
            winning_color = win_filter.iloc[0]['color']
            
            lose_filter = team_info[team_info['school'] == losing_team]
            losing_logo = lose_filter.iloc[0]['logo']
            losing_color = lose_filter.iloc[0]['color'] 
      else:
        
            tgp_result_home = round((tgp_result_home + tgp_result_away_opp) / 2, 2)
            tgp_result_away = round(tgp_result_home * -1, 2)
            
            
            # define winning/losing teams
            if tgp_result_home >= 0:
                winning_team = home
                losing_team = away
                tgp_result_wt = tgp_result_home
            else:
                winning_team = away
                losing_team = home
                tgp_result_wt = tgp_result_away
                              
                              
            # winning and losing team info
            win_filter = team_info[team_info['school'] == winning_team]
            winning_logo = win_filter.iloc[0]['logo']
            winning_color = win_filter.iloc[0]['color']
            
            lose_filter = team_info[team_info['school'] == losing_team]
            losing_logo = lose_filter.iloc[0]['logo']
            losing_color = lose_filter.iloc[0]['color'] 


      return {"tgp_result_home": tgp_result_home, "tgp_result_away": tgp_result_away,
              "winning_team": winning_team, "losing_team": losing_team, "winning_logo": winning_logo,
              "losing_logo": losing_logo, "winning_color": winning_color, "losing_color": losing_color,
              "tgp_result_wt": tgp_result_wt}  
        





# st page width set
st.set_page_config(layout="wide")



# welcome page format
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



# historical betting retults page format
def historical_results_page():
    
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
    col1, col2, col3, col4 = st.columns([3, 1, 8, 5])  

    with col1:
      
        st.write("### Filters")
        
        # Create dropdowns for filtering options on the left column
        team_options = st.selectbox(
            "Team", 
            options=["All"] + unique_teams_sorted,  
            index=0  
        )

        week_options = st.multiselect(
            "Week", 
            options=["All"] + unique_weeks,  
            default=["All"]  
        )

        season_options = st.multiselect(
            "Season", 
            options=["All"] + unique_seasons,  
            default=["All"]  
        )

        pred_home_win_options = st.selectbox(
            "Predicted Home Team Win?", 
            options=["All", "Yes", "No"],  
            index=0  
        )

        actual_home_win_options = st.selectbox(
            "Home Team Actually Won?", 
            options=["All", "Yes", "No"],  
            index=0  
        )
        
        # Filter for pred_home_cover (manual dropdown with 1 or 0)
        pred_home_cover_options = st.selectbox(
            "Predicted Home to Cover?", 
            options=["All", "Yes", "No"],  
            index=0  
        )

        # Filter for book_home_spread (manual number input with bounds)
        # use int and floor/ceiling so the bounds don't round down and exclude outer bound games
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
        # use int and floor/ceiling so the bounds don't round down and exclude outer bound games
        pred_vs_book_lower, pred_vs_book_upper = st.slider(
            "Pred vs Book PD Diff",
            min_value=int(math.floor(df['pred_vs_book_spread'].min())),
            max_value=int(math.ceil(df['pred_vs_book_spread'].max())),
            value=(int(math.floor(df['pred_vs_book_spread'].min())), int(math.ceil(df['pred_vs_book_spread'].max())))
        )
    
    # column for space
    with col2:
        st.write("")
        

    # last column, filtered data frame
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
        
        # these ones are gonna include NAs until the checkboxes filter out
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
        
    
    
    # column with results text
    with col3:
            
            st.write("### Betting Results")

            
            ### SPREAD
            
            # Calculate our return on investment
            our_return_spread = filtered_df['spread_winnings'].sum(skipna=True) / filtered_df['spread_winnings'].count()

            # format it as a percentage and dollars ($100 bet)
            our_return_percentage_spread = f"{(our_return_spread * 100) - 100:.2f}%"  # If you want to show it as a percentage
            our_return_dollars_spread = f"{(100 * our_return_spread):.2f}"
            
            # Calculate bettor average return on investment
            naive_return_spread = filtered_df['naive_spread_winnings'].sum(skipna=True) / filtered_df['naive_spread_winnings'].count()

            # format it as a percentage and dollars ($100 bet)
            naive_return_percentage_spread = f"{(naive_return_spread * 100) - 100:.2f}%"  # If you want to show it as a percentage
            naive_return_dollars_spread = f"{(100 * naive_return_spread):.2f}"
            
            # diff between ours and avg return
            ours_over_naive_spread = f"{100 * our_return_spread - 100 * naive_return_spread:.2f}"
            naive_over_ours_spread = f"{100 * naive_return_spread - 100 * our_return_spread:.2f}"

            
            # format return print
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



            # format return explanations
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
            
            # our return on investment
            our_return_moneyline = filtered_df['ml_winnings'].sum(skipna=True) / filtered_df['ml_winnings'].count()

            # format it as a percentage and dollars ($100 bet)
            our_return_percentage_moneyline = f"{(our_return_moneyline * 100) - 100:.2f}%"  # If you want to show it as a percentage
            our_return_dollars_moneyline = f"{(100 * our_return_moneyline):.2f}"
            
            # average return on investment
            naive_return_moneyline = filtered_df['naive_ml_winnings'].sum(skipna=True) / filtered_df['naive_ml_winnings'].count()

            # format it as a percentage and dollars ($100 bet)
            naive_return_percentage_moneyline = f"{(naive_return_moneyline * 100) - 100:.2f}%"  # If you want to show it as a percentage
            naive_return_dollars_moneyline = f"{(100 * naive_return_moneyline):.2f}"
            
            # diff between ours and average
            ours_over_naive_moneyline = f"{100 * our_return_moneyline - 100 * naive_return_moneyline:.2f}"
            naive_over_ours_moneyline = f"{100 * naive_return_moneyline - 100 * our_return_moneyline:.2f}"

            
            # format overall print          
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



            # format description
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
                
                
                
                
def playoff_page():
  
  # Streamlit App
  st.title("The Playoff")
  
  st.write("Customize your playoff! Select what teams you want in each seed. The model will predict each game, but you can override with the checkbox.")

  st.write("Some notes: first, the underlying data used for these predictions ends at the Conference Championship week. So, it does not currently take recent playoff performance into account.")

  st.write("Another, neutral site games are not yet accounted for by the model. Here, the point differential is calculated by the average of each team playing the other at home.")

  st.write("Also, I know the model loves ND this postseason. I swear it's XGBoost predicting, not me :)")
  
  st.write("  ")


  neutral_site_ind_fr = 0
  neutral_site_ind_rest = 1
  
  # Create columns for layout
  col1, col2, col3, col4, col5, col6, col7, col8, col9 = st.columns([4,2,4,1,4,1,4,1,4])
  
  
  
  with col1:
    
      # Create select boxes with specific default values
      default_values = ["Oregon", "Georgia", "Boise State", "Arizona State", "Texas", "Penn State", 
                      "Notre Dame", "Ohio State", "Tennessee", "Indiana", "SMU", "Clemson"]

      seeds = []
      for value in default_values:
          
        try:
            # Find the index of the default value
            index = theor_prepped["t1_team"].tolist().index(value)
        except ValueError:
            # If the value is not found, use a fallback index (e.g., -1 or 0)
            index = -1  # You can change this to 0 if you want a valid index
        seeds.append(index)
    
      
      # select boxes
      seed1 = st.selectbox("1 Seed", theor_prepped["t1_team"], index = seeds[0])
      seed2 = st.selectbox("2 Seed", theor_prepped["t1_team"], index = seeds[1])
      seed3 = st.selectbox("3 Seed", theor_prepped["t1_team"], index = seeds[2])
      seed4 = st.selectbox("4 Seed", theor_prepped["t1_team"], index = seeds[3])
      seed5 = st.selectbox("5 Seed", theor_prepped["t1_team"], index = seeds[4])
      seed6 = st.selectbox("6 Seed", theor_prepped["t1_team"], index = seeds[5])
      seed7 = st.selectbox("7 Seed", theor_prepped["t1_team"], index = seeds[6])
      seed8 = st.selectbox("8 Seed", theor_prepped["t1_team"], index = seeds[7])
      seed9 = st.selectbox("9 Seed", theor_prepped["t1_team"], index = seeds[8])
      seed10 = st.selectbox("10 Seed", theor_prepped["t1_team"], index = seeds[9])
      seed11 = st.selectbox("11 Seed", theor_prepped["t1_team"], index = seeds[10])
      seed12 = st.selectbox("12 Seed", theor_prepped["t1_team"], index = seeds[11])
      
      
      
  with col3:
    
        
        st.markdown(f"""
            <div style="font-size:30px; font-weight:bold; text-align:center; line-height:1.75; color:orange;">
                1st Round <br>
            </div>
        """, unsafe_allow_html=True)
        
        
        # FR1
        if "override_fr1" not in st.session_state:
            st.session_state.override_fr1 = False
    
        results_fr1 = game_predictor_func(seed8, seed9, neutral_site_ind_fr)
    
        # Display the header for the game
        st.markdown(f"""
            <div style="font-size:20px; font-weight:bold; text-align:center; line-height:2; color:#87CEEB;">
                FR Game 1 <br>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
                <div style="font-size:25px; font-weight:bold; text-align:center; line-height:1;">
                    {seed8} <br> vs <br> {seed9}
                </div>
            """, unsafe_allow_html=True)
        
        # Conditional Display of Matchup based on session_state
        if st.session_state.override_fr1 == True:
            
            st.markdown("<div style='display: flex; justify-content: center;'><img src='" + results_fr1["losing_logo"] + "' width='60'></div>", unsafe_allow_html=True)
            
            st.markdown(f"""
                <div style="font-size:15px; font-weight:bold; color:{results_fr1["winning_color"]}; text-align:center;">
                      
                </div>
            """, unsafe_allow_html=True)
            
            fr1_winner = results_fr1["losing_team"]
            
        else:
            st.markdown("<div style='display: flex; justify-content: center;'><img src='" + results_fr1["winning_logo"] + "' width='60'></div>", unsafe_allow_html=True)
            
            # If override is checked, use a different color or text for the result
            st.markdown(f"""
                <div style="font-size:15px; font-weight:bold; color:{results_fr1["winning_color"]}; text-align:center;">
                    by {results_fr1["tgp_result_wt"]}
                </div>
            """, unsafe_allow_html=True)
            
            fr1_winner = results_fr1["winning_team"]
        
            
        # Place the checkbox at the bottom of the page (after everything else)
        st.checkbox("Override Win?", value=st.session_state.override_fr1, key = "override_fr1")
        
        
        st.write(" ")
        
       
       
        
        # FR2
        if "override_fr2" not in st.session_state:
            st.session_state.override_fr2 = False
    
        results_fr2 = game_predictor_func(seed7, seed10, neutral_site_ind_fr)
    
        # Display the header for the game
        st.markdown(f"""
            <div style="font-size:20px; font-weight:bold; text-align:center; line-height:2; color:#87CEEB;">
                FR Game 2 <br>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
                <div style="font-size:25px; font-weight:bold; text-align:center; line-height:1;">
                    {seed7} <br> vs <br> {seed10}
                </div>
            """, unsafe_allow_html=True)
        
        # Conditional Display of Matchup based on session_state
        if st.session_state.override_fr2 == True:
            
            st.markdown("<div style='display: flex; justify-content: center;'><img src='" + results_fr2["losing_logo"] + "' width='60'></div>", unsafe_allow_html=True)
            
            st.markdown(f"""
                <div style="font-size:15px; font-weight:bold; color:{results_fr2["winning_color"]}; text-align:center;">
                      
                </div>
            """, unsafe_allow_html=True)
            
            fr2_winner = results_fr2["losing_team"]
            
        else:
            st.markdown("<div style='display: flex; justify-content: center;'><img src='" + results_fr2["winning_logo"] + "' width='60'></div>", unsafe_allow_html=True)
            
            # If override is checked, use a different color or text for the result
            st.markdown(f"""
                <div style="font-size:15px; font-weight:bold; color:{results_fr2["winning_color"]}; text-align:center;">
                    by {results_fr2["tgp_result_wt"]}
                </div>
            """, unsafe_allow_html=True)
            
            fr2_winner = results_fr2["winning_team"]
        
            
        # Place the checkbox at the bottom of the page (after everything else)
        st.checkbox("Override Win?", value=st.session_state.override_fr2, key = "override_fr2")
        
        
        st.write(" ")
        
        
        


        # FR3
        if "override_fr3" not in st.session_state:
            st.session_state.override_fr3 = False
    
        results_fr3 = game_predictor_func(seed6, seed11, neutral_site_ind_fr)
    
        # Display the header for the game
        st.markdown(f"""
            <div style="font-size:20px; font-weight:bold; text-align:center; line-height:2; color:#87CEEB;">
                FR Game 3 <br>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
                <div style="font-size:25px; font-weight:bold; text-align:center; line-height:1;">
                    {seed6} <br> vs <br> {seed11}
                </div>
            """, unsafe_allow_html=True)
        
        # Conditional Display of Matchup based on session_state
        if st.session_state.override_fr3 == True:
            
            st.markdown("<div style='display: flex; justify-content: center;'><img src='" + results_fr3["losing_logo"] + "' width='60'></div>", unsafe_allow_html=True)
            
            st.markdown(f"""
                <div style="font-size:15px; font-weight:bold; color:{results_fr3["winning_color"]}; text-align:center;">
                      
                </div>
            """, unsafe_allow_html=True)
            
            fr3_winner = results_fr3["losing_team"]
            
        else:
            st.markdown("<div style='display: flex; justify-content: center;'><img src='" + results_fr3["winning_logo"] + "' width='60'></div>", unsafe_allow_html=True)
            
            # If override is checked, use a different color or text for the result
            st.markdown(f"""
                <div style="font-size:15px; font-weight:bold; color:{results_fr3["winning_color"]}; text-align:center;">
                    by {results_fr3["tgp_result_wt"]}
                </div>
            """, unsafe_allow_html=True)
            
            fr3_winner = results_fr3["winning_team"]
        
            
        # Place the checkbox at the bottom of the page (after everything else)
        st.checkbox("Override Win?", value=st.session_state.override_fr3, key = "override_fr3")
        
        
        st.write(" ")
        
        
        
        
        
        # FR4
        if "override_fr4" not in st.session_state:
            st.session_state.override_fr4 = False
    
        results_fr4 = game_predictor_func(seed5, seed12, neutral_site_ind_fr)
    
        # Display the header for the game
        st.markdown(f"""
            <div style="font-size:20px; font-weight:bold; text-align:center; line-height:2; color:#87CEEB;">
                FR Game 4 <br>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
                <div style="font-size:25px; font-weight:bold; text-align:center; line-height:1;">
                    {seed5} <br> vs <br> {seed12}
                </div>
            """, unsafe_allow_html=True)
        
        # Conditional Display of Matchup based on session_state
        if st.session_state.override_fr4 == True:
            
            st.markdown("<div style='display: flex; justify-content: center;'><img src='" + results_fr4["losing_logo"] + "' width='60'></div>", unsafe_allow_html=True)
            
            st.markdown(f"""
                <div style="font-size:15px; font-weight:bold; color:{results_fr4["winning_color"]}; text-align:center;">
                      
                </div>
            """, unsafe_allow_html=True)
            
            fr4_winner = results_fr4["losing_team"]
            
        else:
            st.markdown("<div style='display: flex; justify-content: center;'><img src='" + results_fr4["winning_logo"] + "' width='60'></div>", unsafe_allow_html=True)
            
            # If override is checked, use a different color or text for the result
            st.markdown(f"""
                <div style="font-size:15px; font-weight:bold; color:{results_fr4["winning_color"]}; text-align:center;">
                    by {results_fr4["tgp_result_wt"]}
                </div>
            """, unsafe_allow_html=True)
            
            fr4_winner = results_fr4["winning_team"]
        
            
        # Place the checkbox at the bottom of the page (after everything else)
        st.checkbox("Override Win?", value=st.session_state.override_fr4, key = "override_fr4")
        
        
        
        
        
        
  with col5:
    
    
        st.markdown(f"""
            <div style="font-size:30px; font-weight:bold; text-align:center; line-height:1.75; color:orange;">
                Quarters <br>
            </div>
        """, unsafe_allow_html=True)


        # Rose Bowl
        if "override_rb" not in st.session_state:
            st.session_state.override_rb = False
    
        results_rb = game_predictor_func(seed1, fr1_winner, neutral_site_ind_rest)
    
        # Display the header for the game
        st.markdown(f"""
            <div style="font-size:20px; font-weight:bold; text-align:center; line-height:2; color:#87CEEB;">
                Rose Bowl <br>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
                <div style="font-size:25px; font-weight:bold; text-align:center; line-height:1;">
                    {seed1} <br> vs <br> {fr1_winner}
                </div>
            """, unsafe_allow_html=True)
        
        # Conditional Display of Matchup based on session_state
        if st.session_state.override_rb == True:
            
            st.markdown("<div style='display: flex; justify-content: center;'><img src='" + results_rb["losing_logo"] + "' width='60'></div>", unsafe_allow_html=True)
            
            st.markdown(f"""
                <div style="font-size:15px; font-weight:bold; color:{results_rb["winning_color"]}; text-align:center;">
                      
                </div>
            """, unsafe_allow_html=True)
            
            rb_winner = results_rb["losing_team"]
            
        else:
            st.markdown("<div style='display: flex; justify-content: center;'><img src='" + results_rb["winning_logo"] + "' width='60'></div>", unsafe_allow_html=True)
            
            # If override is checked, use a different color or text for the result
            st.markdown(f"""
                <div style="font-size:15px; font-weight:bold; color:{results_rb["winning_color"]}; text-align:center;">
                    by {results_rb["tgp_result_wt"]}
                </div>
            """, unsafe_allow_html=True)
            
            rb_winner = results_rb["winning_team"]
        
            
        # Place the checkbox at the bottom of the page (after everything else)
        st.checkbox("Override Win?", value=st.session_state.override_rb, key = "override_rb")
        
        
        st.write(" ")
        
       
       
        
        # Sugar Bowl
        if "override_sb" not in st.session_state:
            st.session_state.override_sb = False
    
        results_sb = game_predictor_func(seed2, fr2_winner, neutral_site_ind_rest)
    
        # Display the header for the game
        st.markdown(f"""
            <div style="font-size:20px; font-weight:bold; text-align:center; line-height:2; color:#87CEEB;">
                Sugar Bowl <br>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
                <div style="font-size:25px; font-weight:bold; text-align:center; line-height:1;">
                    {seed2} <br> vs <br> {fr2_winner}
                </div>
            """, unsafe_allow_html=True)
        
        # Conditional Display of Matchup based on session_state
        if st.session_state.override_sb == True:
            
            st.markdown("<div style='display: flex; justify-content: center;'><img src='" + results_sb["losing_logo"] + "' width='60'></div>", unsafe_allow_html=True)
            
            st.markdown(f"""
                <div style="font-size:15px; font-weight:bold; color:{results_sb["winning_color"]}; text-align:center;">
                      
                </div>
            """, unsafe_allow_html=True)
            
            sb_winner = results_sb["losing_team"]
            
        else:
            st.markdown("<div style='display: flex; justify-content: center;'><img src='" + results_sb["winning_logo"] + "' width='60'></div>", unsafe_allow_html=True)
            
            # If override is checked, use a different color or text for the result
            st.markdown(f"""
                <div style="font-size:15px; font-weight:bold; color:{results_sb["winning_color"]}; text-align:center;">
                    by {results_sb["tgp_result_wt"]}
                </div>
            """, unsafe_allow_html=True)
            
            sb_winner = results_sb["winning_team"]
        
            
        # Place the checkbox at the bottom of the page (after everything else)
        st.checkbox("Override Win?", value=st.session_state.override_sb, key = "override_sb")
        
        
        st.write(" ")
        
        
        


        # Fiesta Bowl
        if "override_fb" not in st.session_state:
            st.session_state.override_fb = False
    
        results_fb = game_predictor_func(seed3, fr3_winner, neutral_site_ind_rest)
    
        # Display the header for the game
        st.markdown(f"""
            <div style="font-size:20px; font-weight:bold; text-align:center; line-height:2; color:#87CEEB;">
                Fiesta Bowl <br>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
                <div style="font-size:25px; font-weight:bold; text-align:center; line-height:1;">
                    {seed3} <br> vs <br> {fr3_winner}
                </div>
            """, unsafe_allow_html=True)
        
        # Conditional Display of Matchup based on session_state
        if st.session_state.override_fb == True:
            
            st.markdown("<div style='display: flex; justify-content: center;'><img src='" + results_fb["losing_logo"] + "' width='60'></div>", unsafe_allow_html=True)
            
            st.markdown(f"""
                <div style="font-size:15px; font-weight:bold; color:{results_fb["winning_color"]}; text-align:center;">
                      
                </div>
            """, unsafe_allow_html=True)
            
            fb_winner = results_fb["losing_team"]
            
        else:
            st.markdown("<div style='display: flex; justify-content: center;'><img src='" + results_fb["winning_logo"] + "' width='60'></div>", unsafe_allow_html=True)
            
            # If override is checked, use a different color or text for the result
            st.markdown(f"""
                <div style="font-size:15px; font-weight:bold; color:{results_fb["winning_color"]}; text-align:center;">
                    by {results_fb["tgp_result_wt"]}
                </div>
            """, unsafe_allow_html=True)
            
            fb_winner = results_fb["winning_team"]
        
            
        # Place the checkbox at the bottom of the page (after everything else)
        st.checkbox("Override Win?", value=st.session_state.override_fb, key = "override_fb")
        
        
        st.write(" ")
        
        
        
        
        
        # Peach Bowl
        if "override_pb" not in st.session_state:
            st.session_state.override_pb = False
    
        results_pb = game_predictor_func(seed4, fr4_winner, neutral_site_ind_rest)
    
        # Display the header for the game
        st.markdown(f"""
            <div style="font-size:20px; font-weight:bold; text-align:center; line-height:2; color:#87CEEB;">
                Peach Bowl <br>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
                <div style="font-size:25px; font-weight:bold; text-align:center; line-height:1;">
                    {seed4} <br> vs <br> {fr4_winner}
                </div>
            """, unsafe_allow_html=True)
        
        # Conditional Display of Matchup based on session_state
        if st.session_state.override_pb == True:
            
            st.markdown("<div style='display: flex; justify-content: center;'><img src='" + results_pb["losing_logo"] + "' width='60'></div>", unsafe_allow_html=True)
            
            st.markdown(f"""
                <div style="font-size:15px; font-weight:bold; color:{results_pb["winning_color"]}; text-align:center;">
                      
                </div>
            """, unsafe_allow_html=True)
            
            pb_winner = results_pb["losing_team"]
            
        else:
            st.markdown("<div style='display: flex; justify-content: center;'><img src='" + results_pb["winning_logo"] + "' width='60'></div>", unsafe_allow_html=True)
            
            # If override is checked, use a different color or text for the result
            st.markdown(f"""
                <div style="font-size:15px; font-weight:bold; color:{results_pb["winning_color"]}; text-align:center;">
                    by {results_pb["tgp_result_wt"]}
                </div>
            """, unsafe_allow_html=True)
            
            pb_winner = results_pb["winning_team"]
        
            
        # Place the checkbox at the bottom of the page (after everything else)
        st.checkbox("Override Win?", value=st.session_state.override_pb, key = "override_pb")
        
        
        
        
  


  with col7:
    
    
        st.markdown(f"""
            <div style="font-size:30px; font-weight:bold; text-align:center; line-height:1.75; color:orange;">
                Semis <br>
            </div>
        """, unsafe_allow_html=True)


        # Cotton Bowl
        if "override_cb" not in st.session_state:
            st.session_state.override_cb = False
    
        results_cb = game_predictor_func(rb_winner, pb_winner, neutral_site_ind_rest)
    
        # Display the header for the game
        st.markdown(f"""
            <div style="font-size:20px; font-weight:bold; text-align:center; line-height:2; color:#87CEEB;">
                Cotton Bowl <br>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
                <div style="font-size:25px; font-weight:bold; text-align:center; line-height:1;">
                    {rb_winner} <br> vs <br> {pb_winner}
                </div>
            """, unsafe_allow_html=True)
        
        # Conditional Display of Matchup based on session_state
        if st.session_state.override_cb == True:
            
            st.markdown("<div style='display: flex; justify-content: center;'><img src='" + results_cb["losing_logo"] + "' width='60'></div>", unsafe_allow_html=True)
            
            st.markdown(f"""
                <div style="font-size:15px; font-weight:bold; color:{results_cb["winning_color"]}; text-align:center;">
                      
                </div>
            """, unsafe_allow_html=True)
            
            cb_winner = results_cb["losing_team"]
            
        else:
            st.markdown("<div style='display: flex; justify-content: center;'><img src='" + results_cb["winning_logo"] + "' width='60'></div>", unsafe_allow_html=True)
            
            # If override is checked, use a different color or text for the result
            st.markdown(f"""
                <div style="font-size:15px; font-weight:bold; color:{results_cb["winning_color"]}; text-align:center;">
                    by {results_cb["tgp_result_wt"]}
                </div>
            """, unsafe_allow_html=True)
            
            cb_winner = results_cb["winning_team"]
        
            
        # Place the checkbox at the bottom of the page (after everything else)
        st.checkbox("Override Win?", value=st.session_state.override_cb, key = "override_cb")
        
        
        st.write(" ")
        
       
       
        
        # Orange Bowl
        if "override_ob" not in st.session_state:
            st.session_state.override_ob = False
    
        results_ob = game_predictor_func(sb_winner, fb_winner, neutral_site_ind_rest)
    
        # Display the header for the game
        st.markdown(f"""
            <div style="font-size:20px; font-weight:bold; text-align:center; line-height:2; color:#87CEEB;">
                Orange Bowl <br>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
                <div style="font-size:25px; font-weight:bold; text-align:center; line-height:1;">
                    {sb_winner} <br> vs <br> {fb_winner}
                </div>
            """, unsafe_allow_html=True)
        
        # Conditional Display of Matchup based on session_state
        if st.session_state.override_ob == True:
            
            st.markdown("<div style='display: flex; justify-content: center;'><img src='" + results_ob["losing_logo"] + "' width='60'></div>", unsafe_allow_html=True)
            
            st.markdown(f"""
                <div style="font-size:15px; font-weight:bold; color:{results_ob["winning_color"]}; text-align:center;">
                      
                </div>
            """, unsafe_allow_html=True)
            
            ob_winner = results_ob["losing_team"]
            
        else:
            st.markdown("<div style='display: flex; justify-content: center;'><img src='" + results_ob["winning_logo"] + "' width='60'></div>", unsafe_allow_html=True)
            
            # If override is checked, use a different color or text for the result
            st.markdown(f"""
                <div style="font-size:15px; font-weight:bold; color:{results_ob["winning_color"]}; text-align:center;">
                    by {results_ob["tgp_result_wt"]}
                </div>
            """, unsafe_allow_html=True)
            
            ob_winner = results_ob["winning_team"]
        
            
        # Place the checkbox at the bottom of the page (after everything else)
        st.checkbox("Override Win?", value=st.session_state.override_ob, key = "override_ob")
        
        
        
        
        


  with col9:
    
        st.markdown(f"""
            <div style="font-size:30px; font-weight:bold; text-align:center; line-height:1.75; color:orange;">
                Natty <br>
            </div>
        """, unsafe_allow_html=True)


        # National Championship
        if "override_nc" not in st.session_state:
            st.session_state.override_nc = False
    
        results_nc = game_predictor_func(cb_winner, ob_winner, neutral_site_ind_rest)
    
        # Display the header for the game
        st.markdown(f"""
            <div style="font-size:20px; font-weight:bold; text-align:center; line-height:2; color:#87CEEB;">
                Championship <br>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
                <div style="font-size:25px; font-weight:bold; text-align:center; line-height:1;">
                    {cb_winner} <br> vs <br> {ob_winner}
                </div>
            """, unsafe_allow_html=True)
        
        # Conditional Display of Matchup based on session_state
        if st.session_state.override_nc == True:
            
            st.markdown("<div style='display: flex; justify-content: center;'><img src='" + results_nc["losing_logo"] + "' width='60'></div>", unsafe_allow_html=True)
            
            st.markdown(f"""
                <div style="font-size:15px; font-weight:bold; color:{results_nc["winning_color"]}; text-align:center;">
                      
                </div>
            """, unsafe_allow_html=True)
            
            nc_winner = results_nc["losing_team"]
            
        else:
            st.markdown("<div style='display: flex; justify-content: center;'><img src='" + results_nc["winning_logo"] + "' width='60'></div>", unsafe_allow_html=True)
            
            # If override is checked, use a different color or text for the result
            st.markdown(f"""
                <div style="font-size:15px; font-weight:bold; color:{results_nc["winning_color"]}; text-align:center;">
                    by {results_nc["tgp_result_wt"]}
                </div>
            """, unsafe_allow_html=True)
            
            nc_winner = results_nc["winning_team"]
        
            
        # Place the checkbox at the bottom of the page (after everything else)
        st.checkbox("Override Win?", value=st.session_state.override_nc, key = "override_nc")
        
        
              

        
        

  
  
  
                
                

# define game predictor page
def game_predictor_page():
  
  
  # Streamlit App
  st.title("Game Predictor")
  
  st.write("Pick a combination of teams and see what the model predicts if they played today!")

  st.write("As a note, neutral site games are not yet accounted for by the model. Here, the point differential is calculated by the average of each team playing the other at home.")
  
  # Create columns for layout
  col1, col2 = st.columns([1,2])
    
  
  # filters
  with col1:
      
      
      st.write("### Filters")
      
      # Step 1: Select Away Team
      away_team = st.selectbox("Select Away Team", theor_prepped["t1_team"], index = 40)
      
      # Step 2: Select Home Team
      home_team = st.selectbox("Select Home Team", theor_prepped["t1_team"], index = 77)
      
      
      # Dropdown menu for Yes or No for neutral site
      neutral_site_input = st.selectbox("Neutral Site?", ["No", "Yes"])
      
      # Encode Yes as 1 and No as 0
      neutral_site_ind = 1 if neutral_site_input == "Yes" else 0
      
      st.write("  ")
      st.write(" ")
      
      st.markdown(f"""
          <div style="font-size:18px; font-weight:bold; text-align:center;">
              Data last updated after... <br> <span style="color: orange;"> Conf Champ Week</span>
          </div>
      """, unsafe_allow_html=True)

      
  
  # column two for predictions
  with col2:  
      
      
      results = game_predictor_func(home_team, away_team, neutral_site_ind)

      
      
      # formatting
      st.markdown(f"""
                            <div style="font-size:35px; font-weight:bold; text-align:center;">
                                {away_team} vs {home_team}...
                            </div>
                        """, unsafe_allow_html=True)
      st.write("  ")
      st.write("  ")
    
      
      st.markdown("<div style='display: flex; justify-content: center;'><img src='" + results["winning_logo"] + "' width='200'></div>", unsafe_allow_html=True)
      
      st.write("  ")
      
      
      if results["tgp_result_home"] >= 0:
                    st.markdown(f"""
                            <div style="font-size:35px; font-weight:bold; color:{results["winning_color"]}; text-align:center;">
                                {home_team} wins by {results["tgp_result_home"]}!
                            </div>
                        """, unsafe_allow_html=True)
      else:
                    st.markdown(f"""
                            <div style="font-size:35px; font-weight:bold; color:{results["winning_color"]}; text-align:center;">
                                {away_team} wins by {results["tgp_result_away"]}!
                            </div>
                        """, unsafe_allow_html=True)
                
       
       
          
      
  

        

    
    
# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ("Home", "Betting Accuracy", "The Playoff", "Game Predictor"))




# Display the selected page
if page == "Home":
    welcome_page()
elif page == "Betting Accuracy":
    historical_results_page()
elif page == "The Playoff":
    playoff_page()
elif page == "Game Predictor":
    game_predictor_page()

    
  

   

# python -m streamlit run cfb_app_launch.py
