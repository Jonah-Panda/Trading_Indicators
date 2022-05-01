#Program that uses the stock data scrapped form trading view to backtest an RSI strategy
import pandas as pd
import mplfinance as mpf
import matplotlib.animation as animation
import os
import numpy as np
import datetime
########################FIX SCATTER PLOT FOR BUY AND SELLS
#Base information
TICKER = 'MSFT'
TIME_FRAME = '1D'
Start_date = '2021-03-4 8:00:00'
End_date = '2021-08-20 13:30:00'

#Indicators
SMA_200 = {'Enable' : 'No',
           'Source' : 'close',
           'Period' : '1D', #['1D', 1(min), 5(min), 15(min), 60(min), 'Same as chart']
           'Length' : 200,
           'Color' : 'lime'} 
SMA_50 = {'Enable' : 'No',
           'Source' : 'close',
           'Period' : 'Same as chart', #['1D', 1(min), 5(min), 15(min), 60(min), 'Same as chart']
           'Length' : 50,
           'Color' : 'teal'} 
SMA_20 = {'Enable' : 'No',
           'Source' : 'close',
           'Period' : 'Same as chart', #['1D', 1(min), 5(min), 15(min), 60(min), 'Same as chart']
           'Length' : 20,
           'Color' : 'hotpink'} 
EMA_20 = {'Enable' : 'No',
         'Length' : 20, 
         'Source' : 'close',
         'Period' : 'Same as chart', #['1D', 1(min), 5(min), 15(min), 60(min), 'Same as chart']
         'Color' : 'red'} 
EMA_50 = {'Enable' : 'No',
         'Length' : 50, 
         'Source' : 'close',
         'Period' : 'Same as chart', #['1D', 1(min), 5(min), 15(min), 60(min), 'Same as chart']
         'Color' : 'blue'}
VWAP = {'Enable' : 'No',
        'Color' : 'hotpink'}
RSI = {'Enable' : 'No',
       'Length' : 14,
       'Source' : 'close',
       'Upper_Band' : 70,
       'Middle_Band' : 50,
       'Lower_Band' : 30}
Bollinger = {'Enable' : 'No',
             'Length' : 20,
             'Period' : 'Same as chart', #['1D', 1(min), 5(min), 15(min), 60(min), 'Same as chart']
             'Source' : 'close',
             'Standard Deviation' : 2}
ADX = {'Enable' : 'No',
       'Length' : 9,
       'Period' : 'Same as chart'} #Keep as same as chart
MACD = {'Enable' : 'No', 
        'Fast Length' : 12,
        'Slow Length' : 26,
        'Source' : 'close',
        'Period' : 'Same as chart', #['1D', 1(min), 5(min), 15(min), 60(min), 'Same as chart']
        'Smoothing' : 9,
        'Osillator MA Type' : 'EMA', #EMA or SMA
        'Signal Line MA Type' : 'EMA'} #EMA or SMA
SMA_EMA_crossover = {'Enable' : 'Yes',
                      'EMA Length' : 20,
                      'SMA Length' : 20,
                      'Source' : 'close',
                      'Period' : 'Same as chart', #['1D', 1(min), 5(min), 15(min), 60(min), 'Same as chart']
                      'Smoothing Length' : 1,
                      'Smoothing Type' : 'SMA', #[EMA, SMA]
                      'Strategy' : 'On'} #[On, Off]

def stripping_closedmarket_data(df, TIME_FRAME):
    market_open = datetime.time(hour=7, minute=30, second=0)
    #Fixes bug in data that the first 30 mins of trading are cut out on the hour mark
    if TIME_FRAME == 60:
        market_open = datetime.time(hour=7, minute=00, second=0)

    market_close = datetime.time(hour=14, minute=00, second=0)
    df['time'] = df.index.time
    df['time'] = df['time'].apply(lambda x: x if (x >= market_open and x < market_close) else np.nan)
    df.dropna(inplace=True)
    df.drop(columns='time', inplace=True)
    return df
def getting_csv(TICKER, TIME_FRAME):
    filename = f'{TICKER}_{TIME_FRAME}.csv'
    df = pd.read_csv("/Users/jonahpandarinath/Desktop/Stock_data/{}/{}".format(TIME_FRAME, filename), index_col=0, parse_dates=True)
    Col_names = df.columns
    print(Col_names)
    if 'index' in Col_names:
        df.drop(columns='index', inplace=True)
    df = stripping_closedmarket_data(df, TIME_FRAME)
    return df
def RSI_indicator(df, RSI_Kwargs):
    df['Change'] = df[RSI_Kwargs['Source']] - df[RSI_Kwargs['Source']].shift(1)
    df['Gain'] = df['Change'].map(lambda x: x if (x >= 0) else 0)
    df['Loss'] = df['Change'].map(lambda x: -x if (x <= 0) else 0)
    df['Avg_gain'] = (df['Gain'].rolling(RSI_Kwargs['Length']).sum()-df['Gain'])/(RSI_Kwargs['Length']-1) #avg of the last rolling 13 days (minus the current one)
    df['Avg_loss'] = (df['Loss'].rolling(RSI_Kwargs['Length']).sum()-df['Loss'])/(RSI_Kwargs['Length']-1) #avg of the last rolling 13 days (minus the current one)
    df['Weighted_RS_gain'] = (((df['Avg_gain']*(RSI_Kwargs['Length']-1))+df['Gain'])/RSI_Kwargs['Length']) #Emphasises the most recent datapoint
    df['Weighted_RS_loss'] = (((df['Avg_loss']*(RSI_Kwargs['Length']-1))+df['Loss'])/RSI_Kwargs['Length']) #Emphasises the most recent datapoint
    df['RSI'] = 100 - (100/(1+(df['Weighted_RS_gain']/df['Weighted_RS_loss'])))
    df.drop(columns=['Change', 'Gain', 'Loss', 'Avg_gain', 'Avg_loss', 'Weighted_RS_gain', 'Weighted_RS_loss'], inplace=True)
    return df
def col_name(indicator, dictionary):
    return f"{indicator}_{dictionary['Length']}_{dictionary['Period']}"
def SMA(dictionary, df):
    #Converts same as chart name to usable time frame
    if dictionary['Period'] == 'Same as chart':
        dictionary['Period'] = TIME_FRAME
    #SMA Calculation
    column_name = col_name('SMA', dictionary)
    if dictionary['Period'] == TIME_FRAME:
        df[column_name] = df[dictionary['Source']].rolling(dictionary['Length']).mean()
    else:
        SMA_df = getting_csv(TICKER, dictionary['Period'])
        SMA_df[column_name] = SMA_df[dictionary['Source']].rolling(dictionary['Length']).mean()
        SMA_df.drop(columns=['open', 'high', 'low', 'close', 'volume'], inplace=True)
        #Merging both dataframes
        df = pd.merge(df, SMA_df, on='date', how='left')
        df[column_name].interpolate(method='pad', inplace=True)
        del SMA_df
    return df
def EMA(dictionary, df):
    #Converts same as chart name to usable time frame
    if dictionary['Period'] == 'Same as chart':
        dictionary['Period'] = TIME_FRAME
    #EMA Calculation
    column_name = col_name('EMA', dictionary)
    if dictionary['Period'] == TIME_FRAME:
        df[column_name] = df[dictionary['Source']].ewm(span=dictionary['Length'], adjust=False).mean()
    else:
        EMA_df = getting_csv(TICKER, dictionary['Period'])
        EMA_df[column_name] = EMA_df[dictionary['Source']].ewm(span=dictionary['Length'], adjust=False).mean()
        EMA_df.drop(columns=['open', 'high', 'low', 'close', 'volume'], inplace=True)
        #Merging both dataframes
        df = pd.merge(df, EMA_df, on='date', how='left')
        df[column_name].interpolate(method='pad', inplace=True)
        del EMA_df
    return df
def Bollinger_Bands(rolling_period, dictionary, df):
    #Converts name same as chart to a number
    if dictionary['Period'] == 'Same as chart':
        dictionary['Period'] = TIME_FRAME
    #Uses downloaded chart for calculations
    if dictionary['Period'] == TIME_FRAME:
        df['Bollinger_mid'] = df[dictionary['Source']].rolling(rolling_period).mean()
        df['Bollinger_Std'] = df[dictionary['Source']].rolling(rolling_period).std()
        df['Bollinger_upper'] = df['Bollinger_mid'] + (df['Bollinger_Std'] * dictionary['Standard Deviation'])
        df['Bollinger_lower'] = df['Bollinger_mid'] - (df['Bollinger_Std'] * dictionary['Standard Deviation'])
    #Uses a different time frame (period) chart for calculations
    else:
        Bollinger_df = getting_csv(TICKER, dictionary['Period'])
        Bollinger_df['Bollinger_mid'] = Bollinger_df[dictionary['Source']].rolling(rolling_period).mean()
        Bollinger_df['Bollinger_Std'] = Bollinger_df[dictionary['Source']].rolling(rolling_period).std()
        Bollinger_df['Bollinger_upper'] = Bollinger_df['Bollinger_mid'] + (Bollinger_df['Bollinger_Std'] * dictionary['Standard Deviation'])
        Bollinger_df['Bollinger_lower'] = Bollinger_df['Bollinger_mid'] - (Bollinger_df['Bollinger_Std'] * dictionary['Standard Deviation'])
        Bollinger_df.drop(columns=['open', 'high', 'low', 'close', 'volume'], inplace=True)
        #Merging both dataframes
        df = pd.merge(df, Bollinger_df, on='date', how='left')
        df['Bollinger_mid'].interpolate(method='pad', inplace=True)
        df['Bollinger_Std'].interpolate(method='pad', inplace=True)
        df['Bollinger_upper'].interpolate(method='pad', inplace=True)
        df['Bollinger_lower'].interpolate(method='pad', inplace=True)
        del Bollinger_df
    return df
def VWAP_indicator(df):
    #Getting csv for the 1 minute timeframe
    VWAP_df = getting_csv(TICKER, 1)
    VWAP_df['Avg_price'] = (VWAP_df['high']+VWAP_df['low']+VWAP_df['close'])/3
    VWAP_df['Dollars_candle'] = VWAP_df['volume']*VWAP_df['Avg_price']
    #Marking dataframe times to separate new trading days
    new_trading_day = datetime.time(hour=7, minute=30, second=0)
    VWAP_df['New_day'] = VWAP_df.index.time
    VWAP_df['New_day'] = VWAP_df['New_day'].apply(lambda x: 'Yes' if (x == new_trading_day) else np.nan)
    #Calculating for VWAP
    VWAP_df['Dollars_today'] = np.nan
    VWAP_df['Volume_today'] = np.nan
    VWAP_df.reset_index(inplace=True)
    for i in range(1, len(VWAP_df)):
        if VWAP_df.loc[i, 'New_day'] == 'Yes':
            VWAP_df.loc[i, 'Dollars_today'] = VWAP_df.loc[i, 'Dollars_candle']
            VWAP_df.loc[i, 'Volume_today'] = VWAP_df.loc[i, 'volume']
        else:
            VWAP_df.loc[i, 'Dollars_today'] = VWAP_df.loc[i-1, 'Dollars_today'] + VWAP_df.loc[i, 'Dollars_candle']
            VWAP_df.loc[i, 'Volume_today'] = VWAP_df.loc[i-1, 'Volume_today'] + VWAP_df.loc[i, 'volume']
        VWAP_df.loc[i, 'VWAP'] = VWAP_df.loc[i, 'Dollars_today'] / VWAP_df.loc[i, 'Volume_today']
    #Merging both dataframes to have VWAP in the master chart
    VWAP_df.drop(columns=['open', 'high', 'low', 'close', 'volume', 'Avg_price', 'Dollars_candle', 'New_day', 'Dollars_today', 'Volume_today'], inplace=True)
    VWAP_df.set_index('date', inplace=True)
    df = pd.merge(df, VWAP_df, on='date', how='left')
    del VWAP_df
    return df
def ADX_indicator(dictionary, df):
    #Converts name same as chart to a number
    if dictionary['Period'] == 'Same as chart':
        dictionary['Period'] = TIME_FRAME
    if dictionary['Period'] == TIME_FRAME:
        #Getting True Range
        df['previous_close'] = df['close'].shift(1)
        df['high-low'] = df['high'] - df['low']
        df['|high-previous_close|'] = abs(df['high'] - df['previous_close'])
        df['|low-previous_close|'] = abs(df['low'] - df['previous_close'])
        df['true_range'] = df[['high-low', '|high-previous_close|', '|low-previous_close|']].values.max(1)
        df.drop(columns=['previous_close', 'high-low', '|high-previous_close|', '|low-previous_close|'], inplace=True)

        #Getting + and - directional movements
        df['previous_high'] = df['high'].shift(1)
        df['previous_low'] = df['low'].shift(1)
        df['high-previous_high'] = df['high'] - df['previous_high']
        df['low-previous_low'] = df['previous_low'] - df['low']
        df['+Directional_movement'] = np.where((df['high-previous_high'] >= df['low-previous_low']), df['high-previous_high'], 0)
        df['-Directional_movement'] = np.where((df['low-previous_low'] > df['high-previous_high']), df['low-previous_low'], 0)
        df.drop(columns=['previous_high', 'previous_low', 'high-previous_high', 'low-previous_low'], inplace=True)

        #Smoothing True range, + Directional movements, and - Directional movements
        df['rolling_true_range'] = df['true_range'].rolling(window=dictionary['Length']).sum().shift(1)
        df['Smooth_true_range'] = df['rolling_true_range'] - (df['rolling_true_range']/dictionary['Length']) + df['true_range']
        df['rolling+DM'] = df['+Directional_movement'].rolling(window=dictionary['Length']).sum().shift(1)
        df['Smooth+DM'] = df['rolling+DM'] - (df['rolling+DM']/dictionary['Length']) + df['+Directional_movement']
        df['rolling-DM'] = df['-Directional_movement'].rolling(window=dictionary['Length']).sum().shift(1)
        df['Smooth-DM'] = df['rolling-DM'] - (df['rolling-DM']/dictionary['Length']) + df['-Directional_movement']
        df.drop(columns=['rolling_true_range', 'rolling+DM', 'rolling-DM'], inplace=True)
        df.drop(columns=['true_range', '+Directional_movement', '-Directional_movement'], inplace=True)

        #+Directional Index Indicator & -Directional Index Indicator
        df['+DI'] = df['Smooth+DM']/df['Smooth_true_range'] * 100
        df['-DI'] = df['Smooth-DM']/df['Smooth_true_range'] * 100
        #DX and ADX
        df['DX'] = abs(df['+DI'] - df['-DI']) / (df['+DI'] + df['-DI']) * 100
        df['ADX'] = df['DX'].rolling(window=dictionary['Length']).mean()
        df.drop(columns=['Smooth+DM', 'Smooth-DM', 'Smooth_true_range', 'DX'], inplace=True)

    return df
def Histogram_colors(Hist_value, previous_hist_value):
    if Hist_value >= 0:
        if previous_hist_value > Hist_value:
            return '#b2dfdb'
        else:
            return '#26a69a'
    else:
        if previous_hist_value > Hist_value:
            return '#ff5252'
        else:
            return '#fecdd2'
def SMA_EMA_crossover_strategy_buy(crossover_line, previous_crossover_line, high, low):
    if crossover_line >= 0 and previous_crossover_line <= 0:
        return low * 0.99
    else:
        return np.nan
def SMA_EMA_crossover_strategy_sell(crossover_line, previous_crossover_line, high, low):
    if crossover_line <= 0 and previous_crossover_line >= 0:
        return high * 1.01
    else:
        return np.nan
def MACD_indicator(dictionary, df):
    #Converts name same as chart to a number
    if dictionary['Period'] == 'Same as chart':
        dictionary['Period'] = TIME_FRAME
    if dictionary['Period'] == TIME_FRAME:
        if dictionary['Osillator MA Type'] == 'EMA':
            df['Fast'] = df[dictionary['Source']].ewm(span=dictionary['Fast Length'], adjust=False).mean()
            df['Slow'] = df[dictionary['Source']].ewm(span=dictionary['Slow Length'], adjust=False).mean()
        else:
            df['Fast'] = df[dictionary['Source']].rolling(dictionary['Fast Length']).mean()
            df['Slow'] = df[dictionary['Source']].rolling(dictionary['Slow Length']).mean()
        df['MACD'] = df['Fast'] - df['Slow']
        if dictionary['Signal Line MA Type'] == 'EMA': 
            df['Signal'] = df['MACD'].ewm(span=dictionary['Smoothing'], adjust=False).mean()
        else:
            df['Signal'] = df['MACD'].rolling(dictionary['Smoothing']).mean()
        df['MACD_histogram'] = df['MACD'] - df['Signal']
        df.drop(columns=['Fast', 'Slow'])

    else:
        MACD_df = getting_csv(TICKER, dictionary['Period'])
        if dictionary['Osillator MA Type'] == 'EMA':
            MACD_df['Fast'] = MACD_df[dictionary['Source']].ewm(span=dictionary['Fast Length'], adjust=False).mean()
            MACD_df['Slow'] = MACD_df[dictionary['Source']].ewm(span=dictionary['Slow Length'], adjust=False).mean()
        else:
            MACD_df['Fast'] = MACD_df[dictionary['Source']].rolling(dictionary['Fast Length']).mean()
            MACD_df['Slow'] = MACD_df[dictionary['Source']].rolling(dictionary['Slow Length']).mean()
        MACD_df['MACD'] = MACD_df['Fast'] - MACD_df['Slow']
        if dictionary['Signal Line MA Type'] == 'EMA': 
            MACD_df['Signal'] = MACD_df['MACD'].ewm(span=dictionary['Smoothing'], adjust=False).mean()
        else:
            MACD_df['Signal'] = MACD_df['MACD'].rolling(dictionary['Smoothing']).mean()
        MACD_df['MACD_histogram'] = MACD_df['MACD'] - MACD_df['Signal']
        
        #Merging both dataframes
        MACD_df.drop(columns=['open', 'high', 'low', 'close', 'volume', 'Fast', 'Slow'], inplace=True)
        df = pd.merge(df, MACD_df, on='date', how='left')
        df['MACD'].interpolate(method='pad', inplace=True)
        df['Signal'].interpolate(method='pad', inplace=True)
        df['MACD_histogram'].interpolate(method='pad', inplace=True)
        del MACD_df
    
    #getting the color mix for the histogram
    df['previous_hist_value'] = df['MACD_histogram'].shift(1)
    df['Histogram_colours'] = df.apply(lambda x: Histogram_colors(x['MACD_histogram'], x['previous_hist_value']),axis=1)
    df.drop(columns=['previous_hist_value'], inplace=True)
    return df
def SMA_EMA_crossover_indicator(dictionary, df):
    #Converts name same as chart to a number
    if dictionary['Period'] == 'Same as chart':
        dictionary['Period'] = TIME_FRAME
    if dictionary['Period'] == TIME_FRAME:
        df['EMA_temp'] = df[dictionary['Source']].ewm(span=dictionary['EMA Length'], adjust=False).mean()
        df['SMA_temp'] = df[dictionary['Source']].rolling(dictionary['SMA Length']).mean()
        df['E&SMA_crossover'] = df['EMA_temp'] - df['SMA_temp']
        if dictionary['Smoothing Type'] == 'EMA': 
            df['E&SMA_crossover'] = df['E&SMA_crossover'].ewm(span=dictionary['Smoothing Length'], adjust=False).mean()
        else:
            df['E&SMA_crossover'] = df['E&SMA_crossover'].rolling(dictionary['Smoothing Length']).mean()
        #df['MACD_histogram'] = df['MACD'] - df['Signal']
        #df.drop(columns=['Fast', 'Slow'])

    else:
        SMA_EMA_crossover_df = getting_csv(TICKER, dictionary['Period'])
        SMA_EMA_crossover_df['EMA_temp'] = SMA_EMA_crossover_df[dictionary['Source']].ewm(span=dictionary['EMA Length'], adjust=False).mean()
        SMA_EMA_crossover_df['SMA_temp'] = SMA_EMA_crossover_df[dictionary['Source']].rolling(dictionary['SMA Length']).mean()
        SMA_EMA_crossover_df['E&SMA_crossover'] = SMA_EMA_crossover_df['EMA_temp'] - SMA_EMA_crossover_df['SMA_temp']
        if dictionary['Smoothing Type'] == 'EMA': 
            SMA_EMA_crossover_df['E&SMA_crossover'] = SMA_EMA_crossover_df['E&SMA_crossover'].ewm(span=dictionary['Smoothing Length'], adjust=False).mean()
        else:
            SMA_EMA_crossover_df['E&SMA_crossover'] = SMA_EMA_crossover_df['E&SMA_crossover'].rolling(dictionary['Smoothing Length']).mean()
        #SMA_EMA_crossover_df['MACD_histogram'] = SMA_EMA_crossover_df['MACD'] - SMA_EMA_crossover_df['Signal']
        
        #Merging both dataframes
        SMA_EMA_crossover_df.drop(columns=['open', 'high', 'low', 'close', 'volume'], inplace=True)
        df = pd.merge(df, SMA_EMA_crossover_df, on='date', how='left')
        df['SMA_temp'].interpolate(method='pad', inplace=True)
        df['EMA_temp'].interpolate(method='pad', inplace=True)
        df['E&SMA_crossover'].interpolate(method='pad', inplace=True)
        del SMA_EMA_crossover_df
    
    #getting the color mix for the histogram
    df['previous_hist_value'] = df['E&SMA_crossover'].shift(1)
    df['Histogram_colours_crossover'] = df.apply(lambda x: Histogram_colors(x['E&SMA_crossover'], x['previous_hist_value']),axis=1)
    df.drop(columns=['previous_hist_value'], inplace=True)
    return df

df = getting_csv(TICKER, TIME_FRAME)
apds = []

#RSI (period adjustable)
if RSI['Enable'] == 'Yes':
    df = RSI_indicator(df, RSI)
    frame = df.loc[Start_date:End_date,:]
    apds.append(mpf.make_addplot(frame['RSI'], panel=1,color='fuchsia',secondary_y=True, width=1))
#20 period SMA
if SMA_20['Enable'] == 'Yes':
    df = SMA(SMA_20, df)
    columns_list = df.columns.tolist()
    frame = df.loc[Start_date:End_date,:]
    apds.append(mpf.make_addplot(frame[columns_list[-1]], color=SMA_20['Color'], width=1))
    del columns_list
#50 period SMA
if SMA_50['Enable'] == 'Yes':
    df = SMA(SMA_50, df)
    columns_list = df.columns.tolist()
    frame = df.loc[Start_date:End_date,:]
    apds.append(mpf.make_addplot(frame[columns_list[-1]], color=SMA_50['Color'], width=1))
    del columns_list
#200 period SMA
if SMA_200['Enable'] == 'Yes':
    df = SMA(SMA_200, df)
    columns_list = df.columns.tolist()
    frame = df.loc[Start_date:End_date,:]
    apds.append(mpf.make_addplot(frame[columns_list[-1]], color=SMA_200['Color'], width=1))
    del columns_list
#20 period EMA
if EMA_20['Enable'] == 'Yes':
    df = EMA(EMA_20, df)
    columns_list = df.columns.tolist()
    frame = df.loc[Start_date:End_date,:]
    apds.append(mpf.make_addplot(frame[columns_list[-1]], color=EMA_20['Color'], width=1, linestyle='solid'))
    del columns_list
#50 period EMA
if EMA_50['Enable'] == 'Yes':
    df = EMA(EMA_50, df)
    columns_list = df.columns.tolist()
    frame = df.loc[Start_date:End_date,:]
    apds.append(mpf.make_addplot(frame[columns_list[-1]], color=EMA_50['Color'], width=1, linestyle='solid'))
    del columns_list
#Bollinger Bands
if Bollinger['Enable'] == 'Yes':
    df = Bollinger_Bands(Bollinger['Length'], Bollinger, df)
    frame = df.loc[Start_date:End_date,:]
    apds.append(mpf.make_addplot(frame['Bollinger_upper'], color='blue', width=1, linestyle='dashdot'))
    apds.append(mpf.make_addplot(frame['Bollinger_lower'], color='blue', width=1, linestyle='dashdot'))
    #apds.append(mpf.make_addplot(frame['Bollinger_mid'], color='orange', width=1, alpha=0.7))
#VWAP
if VWAP['Enable'] == 'Yes':
    df = VWAP_indicator(df)
    frame = df.loc[Start_date:End_date,:]
    apds.append(mpf.make_addplot(frame['VWAP'], color=VWAP['Color'], width=1, linestyle='solid'))
#ADX
if ADX['Enable'] == 'Yes':
    df = ADX_indicator(ADX, df)
    frame = df.loc[Start_date:End_date,:]
    apds.append(mpf.make_addplot(frame['ADX'], panel=1, color='orange',secondary_y=True, width=1))
    apds.append(mpf.make_addplot(frame['+DI'], panel=1, color='lime',secondary_y=True, width=1))
    apds.append(mpf.make_addplot(frame['-DI'], panel=1, color='red',secondary_y=True, width=1))
#MACD
if MACD['Enable'] == 'Yes':
    df = MACD_indicator(MACD, df)
    frame = df.loc[Start_date:End_date,:]
    colors = frame['Histogram_colours'].tolist()
    df.drop(columns=['Histogram_colours'], inplace=True)
    apds.append(mpf.make_addplot(frame['MACD'], panel=1, color='orange',secondary_y=True, width=1))
    apds.append(mpf.make_addplot(frame['Signal'], panel=1, color='blue',secondary_y=True, width=1))
    apds.append(mpf.make_addplot(frame['MACD_histogram'], type='bar', panel=1, color=colors))
#EMA_SMA_crossover
if SMA_EMA_crossover['Enable'] == 'Yes':
    df = SMA_EMA_crossover_indicator(SMA_EMA_crossover, df)
    frame = df.loc[Start_date:End_date,:]
    colors = frame['Histogram_colours_crossover'].tolist()
    df.drop(columns=['Histogram_colours_crossover'], inplace=True)
    apds.append(mpf.make_addplot(frame['EMA_temp'], panel=1, color='orange',secondary_y=True, width=1))
    apds.append(mpf.make_addplot(frame['SMA_temp'], panel=1, color='blue',secondary_y=True, width=1))
    apds.append(mpf.make_addplot(frame['E&SMA_crossover'], type='bar', panel=1, color=colors))
    #Entry and exit strategy
    if SMA_EMA_crossover['Strategy'] == 'On':
        df['previous_hist_value'] = df['E&SMA_crossover'].shift(1)
        df['Buy'] = df.apply(lambda x: SMA_EMA_crossover_strategy_buy(x['E&SMA_crossover'], x['previous_hist_value'], x['high'], x['low']), axis=1)
        df['Sell'] = df.apply(lambda x: SMA_EMA_crossover_strategy_sell(x['E&SMA_crossover'], x['previous_hist_value'], x['high'], x['low']), axis=1)
        df.drop(columns=['previous_hist_value'], inplace=True)
        frame = df.loc[Start_date:End_date,:]
        apds.append(mpf.make_addplot(frame['Buy'], type='scatter', marker = '^', markersize = 80, color='lime'))
        apds.append(mpf.make_addplot(frame['Sell'], type='scatter', marker = 'v', markersize = 80, color='orange'))


#getting the style for the chart
candle_colors = mpf.make_marketcolors(up='#28a49c', down='#f0544c', inherit=True)
trading_view  = mpf.make_mpf_style(base_mpf_style='nightclouds', marketcolors=candle_colors, figcolor='#171b26', facecolor='#171b26', edgecolor='whitesmoke', gridcolor='slategrey', gridstyle='solid' )

frame = df.loc[Start_date:End_date,:]
print(frame.head(5))
mpf.plot(frame, type='candle', addplot=apds, volume = False, style = trading_view)