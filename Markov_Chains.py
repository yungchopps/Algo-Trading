import matplotlib
import matplotlib.pyplot as plt
import io, base64, os, json, re 
import pandas as pd
import numpy as np
import datetime
from random import randint
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns

# build the markov transition grid
def build_transition_grid(compressed_grid, unique_patterns):
    # build the markov transition grid

    patterns = []
    counts = []
    for from_event in unique_patterns:

        # how many times 
        for to_event in unique_patterns:
            pattern = from_event + ',' + to_event # MMM,MlM

            ids_matches = compressed_grid[compressed_grid['Event_Pattern'].str.contains(pattern)]
            found = 0
            if len(ids_matches) > 0:
                Event_Pattern = '---'.join(ids_matches['Event_Pattern'].values)
                found = Event_Pattern.count(pattern)
            patterns.append(pattern)
            counts.append(found)

    # create to/from grid
    grid_Df = pd.DataFrame({'pairs':patterns, 'counts': counts})

    grid_Df['x'], grid_Df['y'] = grid_Df['pairs'].str.split(',', 1).str
    grid_Df.head()

    grid_Df = grid_Df.pivot(index='x', columns='y', values='counts')

    grid_Df.columns= [col for col in grid_Df.columns]
    # del grid_Df.index.name
    grid_Df = grid_Df.rename_axis(None, axis=1)

    # replace all NaN with zeros
    grid_Df.fillna(0, inplace=True)
    grid_Df.head()

    #grid_Df.rowSums(transition_dataframe) 
    grid_Df = grid_Df / grid_Df.sum(1)
    return (grid_Df)



if __name__ == '__main__':
    # load market data from Yahoo Finance (https://finance.yahoo.com/quote/%5EGSPC/history?p=%5EGSPC)

    gspc_df = pd.read_csv('AMD.csv')
    gspc_df['Date'] = pd.to_datetime(gspc_df['Date'])

    cut_off_date = '2010-01-01' 
    gspc_df = gspc_df[gspc_df['Date'] >= cut_off_date]


    gspc_df.head()

    # take random sets of sequential rows 
    new_set = []
    for row_set in range(0, 100000):
        if row_set%2000==0: print(row_set)
        row_quant = randint(10, 30)
        row_start = randint(0, len(gspc_df)-row_quant)
        market_subset = gspc_df.iloc[row_start:row_start+row_quant]

        Close_Date = max(market_subset['Date'])
        if row_set%2000==0: print(Close_Date)
        
        # Close_Gap = (market_subset['Close'] - market_subset['Close'].shift(1)) / market_subset['Close'].shift(1)
        Close_Gap = market_subset['Close'].pct_change()
        High_Gap = market_subset['High'].pct_change()
        Low_Gap = market_subset['Low'].pct_change() 
        Volume_Gap = market_subset['Volume'].pct_change() 
        Daily_Change = (market_subset['Close'] - market_subset['Open']) / market_subset['Open']
        Outcome_Next_Day_Direction = (market_subset['Volume'].shift(-1) - market_subset['Volume'])
        
        new_set.append(pd.DataFrame({'Sequence_ID':[row_set]*len(market_subset),
                                'Close_Date':[Close_Date]*len(market_subset),
                            'Close_Gap':Close_Gap,
                            'High_Gap':High_Gap,
                            'Low_Gap':Low_Gap,
                            'Volume_Gap':Volume_Gap,
                            'Daily_Change':Daily_Change,
                            'Outcome_Next_Day_Direction':Outcome_Next_Day_Direction}))

    new_set_df = pd.concat(new_set)
    print(new_set_df.shape)
    new_set_df = new_set_df.dropna(how='any') 
    print(new_set_df.shape)
    new_set_df.tail(20)


    # create sequences
    # simplify the data by binning values into three groups
    np.seterr(invalid='ignore')
    # Close_Gap
    new_set_df['Close_Gap_LMH'] = pd.qcut(new_set_df['Close_Gap'], 3, labels=["L", "M", "H"])

    # High_Gap - not used in this example
    new_set_df['High_Gap_LMH'] = pd.qcut(new_set_df['High_Gap'], 3, labels=["L", "M", "H"])

    # Low_Gap - not used in this example
    new_set_df['Low_Gap_LMH'] = pd.qcut(new_set_df['Low_Gap'], 3, labels=["L", "M", "H"])

    # Volume_Gap
    new_set_df['Volume_Gap_LMH'] = pd.qcut(new_set_df['Volume_Gap'], 3, labels=["L", "M", "H"])
    
    # Daily_Change
    new_set_df['Daily_Change_LMH'] = pd.qcut(new_set_df['Daily_Change'], 3, labels=["L", "M", "H"])

    # new set
    new_set_df = new_set_df[["Sequence_ID", 
                            "Close_Date", 
                            "Close_Gap_LMH", 
                            "Volume_Gap_LMH", 
                            "Daily_Change_LMH", 
                            "Outcome_Next_Day_Direction"]]

    new_set_df['Event_Pattern'] = new_set_df['Close_Gap_LMH'].astype(str) + new_set_df['Volume_Gap_LMH'].astype(str) + new_set_df['Daily_Change_LMH'].astype(str)


    # reduce the set
    compressed_set = new_set_df.groupby(['Sequence_ID', 
                                        'Close_Date'])['Event_Pattern'].apply(lambda x: "{%s}" % ', '.join(x)).reset_index()

    print(compressed_set.shape)
    compressed_set.head()

    #compressed_outcomes = new_set_df[['Sequence_ID', 'Close_Date', 'Outcome_Next_Day_Direction']].groupby(['Sequence_ID', 'Close_Date']).agg()

    compressed_outcomes = new_set_df.groupby(['Sequence_ID', 'Close_Date'])['Outcome_Next_Day_Direction'].mean()
    compressed_outcomes = compressed_outcomes.to_frame().reset_index()
    print(compressed_outcomes.shape)
    compressed_outcomes.describe()
 
    compressed_set = pd.merge(compressed_set, compressed_outcomes, on= ['Sequence_ID', 'Close_Date'], how='inner')
    print(compressed_set.shape)
    compressed_set.head()

    # # reduce set 

    # compressed_set = new_set_df.groupby(['Sequence_ID', 'Close_Date','Outcome_Next_Day_Direction'])['Event_Pattern'].apply(lambda x: "{%s}" % ', '.join(x)).reset_index()

    compressed_set['Event_Pattern'] = [''.join(e.split()).replace('{','')
                                    .replace('}','') for e in compressed_set['Event_Pattern'].values]
    compressed_set.head()

    # use last x days of data for validation
    compressed_set_validation = compressed_set[compressed_set['Close_Date'] >= datetime.datetime.now() 
                                            - datetime.timedelta(days=90)] # Sys.Date()-90 

    compressed_set_validation.shape

    compressed_set = compressed_set[compressed_set['Close_Date'] < datetime.datetime.now() 
                                           - datetime.timedelta(days=90)]  
    compressed_set.shape

    list(compressed_set)

    # drop date field
    compressed_set = compressed_set[['Sequence_ID', 'Event_Pattern','Outcome_Next_Day_Direction']]
    compressed_set_validation = compressed_set_validation[['Sequence_ID', 'Event_Pattern','Outcome_Next_Day_Direction']]

    compressed_set['Outcome_Next_Day_Direction'].describe()

    print(len(compressed_set['Outcome_Next_Day_Direction']))
    len(compressed_set[abs(compressed_set['Outcome_Next_Day_Direction']) > 150000])

    # keep only keep big/interesting moves 
    print('all moves:', len(compressed_set))
    compressed_set = compressed_set[abs(compressed_set['Outcome_Next_Day_Direction']) > 150000]
    compressed_set['Outcome_Next_Day_Direction'] = np.where((compressed_set['Outcome_Next_Day_Direction'] > 0), 1, 0)
    compressed_set_validation['Outcome_Next_Day_Direction'] = np.where((compressed_set_validation['Outcome_Next_Day_Direction'] > 0), 1, 0)
    print('big moves only:', len(compressed_set))

    compressed_set.head()

    # create two data sets - won/not won
    compressed_set_pos = compressed_set[compressed_set['Outcome_Next_Day_Direction']==1][['Sequence_ID', 'Event_Pattern']]
    print(compressed_set_pos.shape)
    compressed_set_neg = compressed_set[compressed_set['Outcome_Next_Day_Direction']==0][['Sequence_ID', 'Event_Pattern']]
    print(compressed_set_neg.shape)

    flat_list = [item.split(',') for item in compressed_set['Event_Pattern'].values ]
    unique_patterns = ','.join(str(r) for v in flat_list for r in v)
    unique_patterns = list(set(unique_patterns.split(',')))
    len(unique_patterns)

    compressed_set['Outcome_Next_Day_Direction'].tail()

    grid_pos = build_transition_grid(compressed_set_pos, unique_patterns) 
    grid_neg = build_transition_grid(compressed_set_neg, unique_patterns)

    # predict on out of sample data
    actual = []
    predicted = []
    for seq_id in compressed_set_validation['Sequence_ID'].values:
        patterns = compressed_set_validation[compressed_set_validation['Sequence_ID'] == seq_id]['Event_Pattern'].values[0].split(',')
        pos = []
        neg = []
        log_odds = []
        
        for id in range(0, len(patterns)-1):
            # get log odds
            # logOdds = log(tp(i,j) / tn(i,j)
            if (patterns[id] in list(grid_pos) and patterns[id+1] in list(grid_pos) and patterns[id] in list(grid_neg) and patterns[id+1] in list(grid_neg)):
                    
                numerator = grid_pos[patterns[id]][patterns[id+1]]
                denominator = grid_neg[patterns[id]][patterns[id+1]]
                if (numerator == 0 and denominator == 0):
                    log_value =0
                elif (denominator == 0):
                    log_value = np.log(numerator / 0.00001)
                elif (numerator == 0):
                    log_value = np.log(0.00001 / denominator)
                else:
                    log_value = np.log(numerator/denominator)
            else:
                log_value = 0
            
            log_odds.append(log_value)
            
            pos.append(numerator)
            neg.append(denominator)
        
        # print('outcome:', compressed_set_validation[compressed_set_validation['Sequence_ID']==seq_id]['Outcome_Next_Day_Direction'].values[0])
        # print(sum(pos)/sum(neg))
        # print(sum(log_odds))

        actual.append(compressed_set_validation[compressed_set_validation['Sequence_ID']==seq_id]['Outcome_Next_Day_Direction'].values[0])
        predicted.append(sum(log_odds))

    confusion_matrix(actual, [1 if p > 0 else 0 for p in predicted])

    score = accuracy_score(actual, [1 if p > 0 else 0 for p in predicted])
    print('Accuracy:', round(score * 100,2), '%')

    cm = confusion_matrix(actual, [1 if p > 0 else 0 for p in predicted])
    fig, ax = plt.subplots(figsize=(5, 5))
    sns.heatmap(cm, annot=True, ax = ax, fmt='g')

    ax.set_title('Confusion Matrix') 
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')

    ax.xaxis.set_ticklabels(['up day','down day'])
    ax.yaxis.set_ticklabels(['up day','down day'])
    ax.set_yticklabels(ax.get_yticklabels(), rotation = 0, fontsize = 8)
    ax.set_xticklabels(ax.get_xticklabels(), rotation = 90, fontsize = 8)  
    plt.show()



