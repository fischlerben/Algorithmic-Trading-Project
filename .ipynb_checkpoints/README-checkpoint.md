# Algorithmic Trading with the Fama-French Three and Five-Factor Models
![Algo](https://cdn.corporatefinanceinstitute.com/assets/fama-french-three-factor-model02.png)

This project set out to examine the Fama-French Three and Five-Factor Models in order to determine which model resulted in the greatest profit.

MORE

Team members include Ben Fischler, Zachary Green, Priya Roy, Frank Xu and Emmanuel Henao.

---

## Fama-French Three-Factor Model:
In order to more seamlessly progress through the project, we defined various functions that would allow the team to easily read in a CSV and return a cleaned DataFrame:

    def get_factors(factors):
      factor_file = factors + ".csv"
      factor_df = pd.read_csv(factor_file)
      factor_df = factor_df.rename(columns={'Unnamed: 0': 'Date'})

      factor_df['Date'] = factor_df['Date'].apply(lambda x: pd.to_datetime(str(x), format='%Y%m%d'))
      factor_df = factor_df.set_index('Date')

      return factor_df
      
With this "get_factors" function, we were then able to quickly read in any dataframe and not have to manually clean it each time.  Using this function resulted in the following Fama-French Three-Factor dataframe from which we were able to work off:

![three_factor](/Screenshots/three_factor.png?raw=true)

The next function we defined allowed us to the very same thing, but this time, with numbers related to a specific stock's returns as opposed to Fama-French factors:

    def choose_stock(ticker):
      ticker_file=ticker+".csv"
      stock=pd.read_csv(ticker_file, index_col='Date', parse_dates=True, infer_datetime_format=True)
      stock["Returns"]=stock["Close"].dropna().pct_change()*100
      stock.index = pd.Series(stock.index).dt.date

      return stock
      
With this "choose_stock" function, we were then able to quickly read in any dataframe and not have to manually clean it each time.  Using this function resulted in the following ATT dataframe from which we were able to work off:

![att](/Screenshots/att.png?raw=true)

Next, we concatenated the above two dataframes so that we had all of the factors we wanted to use (the 3 Fama-French factors), as well as the target column (Returns) we wanted to use, in the same dataframe:

    combined_df = pd.concat([factors, stock], axis='columns', join='inner')
    combined_df = combined_df.dropna()
    combined_df = combined_df.drop('RF', axis=1)

This gave us the following DataFrame: Mkt-RF, SMB and HML as the 3 factors, and Returns as the target:

![concat_df](/Screenshots/concat_df.png?raw=true)

We were now prepared to define X/y variables, split into Training/Testing data, and run our Linear Regression using SK Learn:

    from sklearn.linear_model import LinearRegression
    lin_reg_model = LinearRegression(fit_intercept=True)
    lin_reg_model = lin_reg_model.fit(X_train, y_train)
    predictions = lin_reg_model.predict(X_test)

We converted y_test to a dataframe using ".to_frame()" and added these predictions to the y_test dataframe.  Now came time to generate the signals we would use to backtest our trading strategy, based on the predictions obtained from the 3 Fama-French factors: we decided to "buy" when the day's predicted returns were greater than the day's actual returns, and "sell" when the opposite was true:

    y_test['Buy Signal'] = np.where(y_test['Predictions'] > y_test['Returns'], 1.0,0.0)
    
Now that we had all of this information together in one dataframe, we were able to build on what was already in there, adding new columns showing portfolio performance as time went on.  Again, we defined a function to make it easier to change dataframe working off of, starting capital and share count:

    def generate_signals(input_df, start_capital=100000, share_count=2000):
      initial_capital = float(start_capital)
      signals_df = input_df.copy()
      share_size = share_count

      # Take a 500 share position where the Buy Signal is 1 (day's predictions greater than day's returns):
      signals_df['Position'] = share_size * signals_df['Buy Signal']

      # Make Entry / Exit Column:
      signals_df['Entry/Exit']=signals_df["Buy Signal"].diff()

      # Find the points in time where a 500 share position is bought or sold:
      signals_df['Entry/Exit Position'] = signals_df['Position'].diff()

      # Multiply share price by entry/exit positions and get the cumulative sum:
      signals_df['Portfolio Holdings'] = signals_df['Close'] * signals_df['Entry/Exit Position'].cumsum()

      # Subtract the initial capital by the portfolio holdings to get the amount of liquid cash in the portfolio:
      signals_df['Portfolio Cash'] = initial_capital - (signals_df['Close'] * signals_df['Entry/Exit Position']).cumsum()

      # Get the total portfolio value by adding the cash amount by the portfolio holdings (or investments):
      signals_df['Portfolio Total'] = signals_df['Portfolio Cash'] + signals_df['Portfolio Holdings']

      # Calculate the portfolio daily returns:
      signals_df['Portfolio Daily Returns'] = signals_df['Portfolio Total'].pct_change()

      # Calculate the cumulative returns:
      signals_df['Portfolio Cumulative Returns'] = (1 + signals_df['Portfolio Daily Returns']).cumprod() - 1

      signals_df = signals_df.dropna()
  
      return signals_df

Using this function, we generated our backtested portfolio returns for y_test.  This resulted in the following dataframe, which shows important numbers like Portfolio Holdings and Cumulative Portfolio Returns:

![signals_df](/Screenshots/signals_df.png?raw=true)

Next came time to evaluate the algorithm, and again, we utiliized a custom function to easily calculate different metrics based on which dataframe is read-in:

    def algo_evaluation(signals_df):
      # Prepare DataFrame for metrics
      metrics = ['Annual Return', 'Cumulative Returns', 'Annual Volatility', 'Sharpe Ratio', 'Sortino Ratio']

      columns = ['Backtest']

      # Initialize the DataFrame with index set to evaluation metrics and column as `Backtest`:
      portfolio_evaluation_df = pd.DataFrame(index=metrics, columns=columns)
      
      # Calculate cumulative returns:
      portfolio_evaluation_df.loc['Cumulative Returns'] = signals_df['Portfolio Cumulative Returns'][-1]
      
      # Calculate annualized returns:
      portfolio_evaluation_df.loc['Annual Return'] = (signals_df['Portfolio Daily Returns'].mean() * 252)
      
      # Calculate annual volatility:
      portfolio_evaluation_df.loc['Annual Volatility'] = (signals_df['Portfolio Daily Returns'].std() * np.sqrt(252))
      
      # Calculate Sharpe Ratio:
      portfolio_evaluation_df.loc['Sharpe Ratio'] = (signals_df['Portfolio Daily Returns'].mean() * 252) / (signals_df['Portfolio Daily Returns'].std() * np.sqrt(252))

      #Calculate Sortino Ratio/Downside Return:
      sortino_ratio_df = signals_df[['Portfolio Daily Returns']].copy()
      sortino_ratio_df.loc[:,'Downside Returns'] = 0
      target = 0
      mask = sortino_ratio_df['Portfolio Daily Returns'] < target
      sortino_ratio_df.loc[mask, 'Downside Returns'] = sortino_ratio_df['Portfolio Daily Returns']**2
      down_stdev = np.sqrt(sortino_ratio_df['Downside Returns'].mean()) * np.sqrt(252)
      expected_return = sortino_ratio_df['Portfolio Daily Returns'].mean() * 252
      sortino_ratio = expected_return/down_stdev
      portfolio_evaluation_df.loc['Sortino Ratio'] = sortino_ratio

      return portfolio_evaluation_df

Using this function, we were able to generate an evaluation table for any dataframe we read-in.  This is the one for ATT, based on the signals dataframe generated above:

![metrics](/Screenshots/metrics.png?raw=true)

Then we generated the same metrics, but this time, without using our backtested strategy of buy-and-sell; this time, we generated the metrics based on a buy-and-hold strategy to compare how our model did vs. how it would have done should it have simply held throughout:

![metrics_with_underlying](/Screenshots/metrics_with_underlying.png?raw=true)

Last step was to define a function which accepted the daily signals dataframe and returned evaluations of individual trades:

    def trade_evaluation(signals_df):
      trade_evaluation_df = pd.DataFrame(
        columns=['Entry Date', 'Exit Date', 'Shares', 'Entry Share Price', 'Exit Share Price', 'Entry Portfolio Holding', 'Exit Portfolio Holding', 'Profit/Loss'])
  
      entry_date = ''
      exit_date = ''
      entry_portfolio_holding = 0
      exit_portfolio_holding = 0
      share_size = 0
      entry_share_price = 0
      exit_share_price = 0

      # Loop through signal DataFrame. If `Entry/Exit` is 1, set entry trade metrics.  Else if `Entry/Exit` is -1, set exit trade metrics and calculate profit, then append the record to the trade evaluation DataFramefor index, row in signals_df.iterrows():
          if row['Entry/Exit'] == 1:
              entry_date = index
              entry_portfolio_holding = row['Portfolio Total']
              share_size = row['Entry/Exit Position']
              entry_share_price = row['Close']

          elif row['Entry/Exit'] == -1:
              exit_date = index
              exit_portfolio_holding = abs(row['Portfolio Total'])
              exit_share_price = row['Close']
              profit_loss = exit_portfolio_holding - entry_portfolio_holding
              trade_evaluation_df = trade_evaluation_df.append(
                  {
                      'Entry Date': entry_date, 'Exit Date': exit_date, 'Shares': share_size, 'Entry Share Price': entry_share_price, 'Exit Share Price': exit_share_price, 'Entry Portfolio Holding': entry_portfolio_holding, 'Exit Portfolio Holding': exit_portfolio_holding, 'Profit/Loss': profit_loss}, ignore_index=True)

      return trade_evaluation_df
      
This allowed us to generate the following evaluation dataframe by feeding in the above-generated signals_df.  It calculates important numbers such as Entry/Exit Share Prices and Profit/Loss, all per trade:

![final_metrics](/Screenshots/final_metrics.png?raw=true)
      
### 3-Factor Results:
ANOVA tables were generated utilizing statsmodels' Ordinary Least Squares (OLS) model, a model analogous to Linear Regressions.  The following is the ANOVA table generated for ATT's 3-Factor results:
![att_3_factor_OLS](/Screenshots/att_3_factor_OLS.png?raw=true)

Few of the important numbers we'd like to point out: the first is r-squared.  The way to read this is that 43.6% of the variation in y, which was excess return on the market, is explained by the factors.  As there were only 3 factors, we expected to see that the 5 factor fama French model might yield better results.  

The prob (f-statistic) depicts probability of the null hypothesis being true, and  can be thought of as the p-value for the regression as a whole.  Our f-statistic of near-0 implies that overall, the regressions were meaningful.   

Last thing are the coefficients and the p-values for the X variables.  The coefficients tell you the size of the effect that variable is having on the dependent variable, when all other independent variables are held constant.  We can see out of all 3 variables, Mkt-Rf has the greatest impact on Returns, which makes intuitive sense as Mkt-Rf is directly used to calculate Returns.  In regards to the p-values for the individual variables: these variables are correlated, and we see that a couple of the variables were deemed insignificant by the model, while the overall p-value was significant.  We reasoned that this is because there is multicollinearity going on with the variables that are explaining the same part of the variation in returns, so their "significance" is divided up among them.

Seaborn Pair Plots were also generated for each stock.  This plot shows the affect of all X variables when all of the other X variables are held constant.  It acts as a visualization of the variable coefficients: the higher the coefficient, the more severe the slope of the line will be, either positively or negatively correlated.  This is the Seaborn Pair Plot for ATT's Three-Factor model:
![att_3_factor_plot](/Screenshots/att_3_factor_plot.png?raw=true)
With Mkt-Rf on the X-axis, we see that as it goes up so do returns, which makes sense, as it had the highest coeffiecient.  The other two variables had slight effects on the dependent variable.

![att_results_plot](/Screenshots/att_results_plot.png?raw=true)
      
The above plot was generated for each stock analyzed: ATT, DIS and SPY, this one being for ATT.  It is a visual representation of the 3-factor algorithm's cumulative returns vs. the underlying, aka buy-and-hold, return.  We see in the above plot that for ATT, our backtested strategy signficantly outperformed a simple buy-and-hold strategy; while our backtested strategy had positive returns, if you were to buy-and-hold, you would have lost money.    
      
---

## Fama-French Five-Factor Model:
Now it was time to do the same as we just did, but this time, using the Fama-French Five-Factor Model rather than the Three-Factor model.  The 2 additional factors are RMW (Robust Minus Weak returns, aka the Profitability Factor) and CMA (Conservative Minus Aggressive returns, aka the Investment Factor).  Given that there were 2 additional factors added for which the model could utilize to help make predictions, the group anticipated that these results would be significantly better than those we had observed for the Three-Factor model.

This is where our custom functions came in handy: we used our pre-defined functions to read-in the new Five-Factor dataframe and perform all the necessary calculations and evaluations.  New code was not needed to be written.  Therefore, we can jump straight to results for the Five-Factor Model:

### 5-Factor Results:
Just like for the Three-Factor Model, ANOVA tables were generated utilizing statsmodels' Ordinary Least Squares (OLS) model, a model analogous to Linear Regressions.  The following is the ANOVA table generated for ATT's 5-Factor results:
![att_5_factor_OLS](/Screenshots/att_5_factor_OLS.png?raw=true)

R-squared increased only slightly, from 43.6% to 46.4%, and the f-statistic was already near-0 with 3 factor.  In terms of coefficients and individual p-values, Mkt-Rf remained the most influential variable, while the one of the new variables, RMW, was the 2nd most influential.  Again, we saw multicollinearity among the X-variables.  Overall, adding these 2 additional factors did not have a significant impact on the model.

![att_5_factor_plot](/Screenshots/att_5_factor_plot.png?raw=true)
Similar to the Three-Factor model, we see that Mkt-Rf is the most influential stock in either direction, positive or negative.  With more factors added, these charts were less steep than when there are only 3 factors because with more relevant factors added, each one now inherently has less of an effect on returns.

![att_5_results](/Screenshots/att_5_results.png?raw=true)
      
We see in the above plot that for ATT, our backtested strategy signficantly outperformed a simple buy-and-hold strategy; while our backtested strategy had positive returns, if you were to buy-and-hold, you would have lost money.  Interestingly, total cumulative returns for the Five-Factor Model were less than those of the Three-Factor Model.  
      
---

## Implications/Conclusions:
