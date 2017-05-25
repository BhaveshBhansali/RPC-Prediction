import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import statsmodels.formula.api as smf
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR



def data_summary(data):
    """Summary of Data

      Parameters:
      -----------
      data : pandas DataFrame with all variables

      response: print column names, dimensions, first 20 lines, statistics (mean, std., min, max) of data

      Returns:
      --------
      Null
      """
    print("Columns of Dataset")
    print(data.columns)

    print("Shape of Data")
    print(data.shape)

    print("First 20 lines")
    print(data.head(20))

    print("Statistics of Data")
    print(data.describe())


def data_exploration_analysis(train_data,prediction_data):
    """Data exploration and visualization

         Parameters:
         -----------
         data : pandas DataFrames of training and prediction data

         response:  call data_summary() for training and prediction data,
                    print total sum of clicks and revenue of training data,
                    create a flag for Train and Prediction Dataset,
                    Concatenate Train and Prediction Data to do preprocessing all together,
                    Visaualize Training Data via Histogram


         Returns:
         --------
         Concatenation of training and prediction data with a flag value to distinguish train and prediction data
    """
    data_summary(train_data)
    data_summary(prediction_data)


    print(str(train_data['Clicks'].sum()))
    print(str(train_data['Revenue'].sum()))

    train_data['Type'] = 'Train'
    prediction_data['Type'] = 'Predict'
    full_data = pd.concat([train_data, prediction_data], axis=0)

    print(full_data.head(10))
    print(full_data.tail(10))

    full_data[full_data['Type']=='Train'].hist()
    plt.show()

    return full_data


def data_preprocessing(full_data,cat_cols):
    """Data Preprocessing

          Parameters:
          -----------
          data : pandas DataFrames of training data and list of category variables

          response:  To check if any column with missing values,
                     Fill null values with 0(can impute mean or some other derived values according to business logic, in this case we do not have null values training data),
                     create a target variable (RPC),
                     Calling data)summary function to look into statistics of RPC variable of training data,
                     look into other statistics like number of rows with RPC=0.0, total sum of RPC,
                     create label encoders for categorical variables,
                     Convert categorical variable into dummy variables


          Returns:
          --------
          full data (concatenation of train and prediction) with newly created target variable 'RPC',
          encoding of category varibles values, replacing NaN and dummy variables for Device_ID and Match_type_ID
     """
    print(full_data.isnull().any())
    full_data = full_data.fillna(0)

    full_data['RPC'] = full_data['Revenue'] / full_data['Clicks']

    data_summary(full_data[full_data['Type'] == 'Train'])

    print(str(len(full_data[(full_data['Type'] == 'Train') & (full_data['RPC'] == 0.0)]['RPC'])))
    print(str(full_data[(full_data['Type'] == 'Train')]['RPC'].sum()))

    for var in cat_cols:
        number = LabelEncoder()
        full_data[var] = number.fit_transform(full_data[var])

    full_data=pd.get_dummies(full_data,columns=['Device_ID','Match_type_ID'])
    return full_data


def forward_selected(data, response):
    """Linear model designed using forward selection.

    Parameters:
    -----------
    data : pandas DataFrame with all possible predictors and response

    response: string, name of response column in data

    Returns:
    --------
    model: an "optimal" fitted statsmodels linear model
           with an intercept
           selected by forward selection
           evaluated by adjusted R-squared
    """
    remaining = set(data.columns)
    remaining.remove(response)
    selected = []
    current_score, best_new_score = 0.0, 0.0
    while remaining and current_score == best_new_score:
        scores_with_candidates = []
        for candidate in remaining:
            formula = "{} ~ {} + 1".format(response,' + '.join(selected + [candidate]))
            score = smf.ols(formula, data).fit().rsquared_adj
            scores_with_candidates.append((score, candidate))
        scores_with_candidates.sort()
        best_new_score, best_candidate = scores_with_candidates.pop()
        if current_score < best_new_score:
            remaining.remove(best_candidate)
            selected.append(best_candidate)
            current_score = best_new_score
    formula = "{} ~ {} + 1".format(response,
                                   ' + '.join(selected))
    model = smf.ols(formula, data).fit()
    return model



def linear_regression(x_train,y_train,x_validate,y_validate):
    """Linear regression model

        Parameters:
        -----------
        data : training and validation data with their response
        response: Predictions, RSquare score and mean square error on validation data

        Returns:
        --------
        mean square error, RSquare Score, model
        """
    model = LinearRegression(fit_intercept=True,normalize=False,copy_X=True,n_jobs=1)
    model.fit(x_train, y_train)
    predictions = model.predict(x_validate)
    mean_square_error=mean_squared_error(predictions, y_validate)
    RSquare=model.score(x_validate,y_validate)

    return mean_square_error, RSquare, model



def polynomial_regression(x_train,y_train,x_validate,y_validate):
    """Polynomial regression model

           Parameters:
           -----------
           data : training and validation data with their response
           response: Predictions, RSquare score and mean square error on validation data

           Returns:
           --------
           mean square error and RSquare Score
           """
    poly = PolynomialFeatures(degree=3)

    x_train = poly.fit_transform(x_train)
    x_validate = poly.fit_transform(x_validate)

    model = LinearRegression(fit_intercept=True,normalize=False,copy_X=True,n_jobs=1)
    model.fit(x_train, y_train)
    predictions = model.predict(x_validate)
    mean_square_error = mean_squared_error(predictions, y_validate)
    RSquare = model.score(x_validate, y_validate)

    return mean_square_error, RSquare


def randomforest_regressor(x_train,y_train,x_validate,y_validate):
    """RandomForest regression model

            Parameters:
            -----------
            data : training and validation data with their response
            response: Predictions, RSquare score and mean square error on validation data

            Returns:
            --------
            mean square error and RSquare Score
            """

    model = RandomForestRegressor(n_estimators=100, min_samples_leaf=10, random_state=1)
    model.fit(x_train, y_train)
    predictions = model.predict(x_validate)
    mean_square_error = mean_squared_error(predictions, y_validate)
    RSquare = model.score(x_validate, y_validate)

    return mean_square_error, RSquare


def support_vector_regression(x_train,y_train,x_validate,y_validate):
    """Suppoert Vector Machine regression model

            Parameters:
            -----------
            data : training and validation data with their response
            response: Predictions, RSquare score and mean square error on validation data

            Returns:
            --------
            mean square error and RSquare Score
            """
    model = SVR(C=1.0,epsilon=0.2)
    model.fit(x_train, y_train)
    predictions = model.predict(x_validate)
    mean_square_error = mean_squared_error(predictions, y_validate)
    RSquare = model.score(x_validate, y_validate)

    return mean_square_error, RSquare


def train_grouping(Train,grouping_features_list):
    """grouping data based on same features

      Parameters:
      -----------
      data : pandas DataFrame with training data and feature list for grouping data

      response: grouping data of same features and compute revenue per click corresponding group

      Returns:
      --------
      training data group by features and new RPC on grouping
      """

    Train = Train.groupby(by=grouping_features_list, sort=False, group_keys=False, squeeze=True)
    print(len(Train))
    x_train = []
    y_train = []
    for name, group in Train:
        df = pd.DataFrame(data=group)
        x_train.append(list(name))
        y_train.append([df['Revenue'].sum() / df['Clicks'].sum()])

    x_train = np.array(x_train)
    y_train = np.array(y_train)

    return x_train,y_train



def main():

    # Load Dataset
    train_data=pd.read_csv('train.csv',dtype={'Account_ID':object,'Ad_group_ID':object,'Campaign_ID':object,'Device_ID':object,'Keyword_ID':object,'Match_type_ID':object})
    prediction_data=pd.read_csv('prediction.csv',dtype={'Account_ID':object,'Ad_group_ID':object,'Campaign_ID':object,'Device_ID':object,'Keyword_ID':object,'Match_type_ID':object})

    # Number of unique Keyword_ID, Ad_group_ID, Campaign_ID, Account_Id, Match_type_ID, Device_ID from training data to check number of encodings of these variables ahead
    print(len(train_data['Keyword_ID'].unique()))
    print(len(train_data['Ad_group_ID'].unique()))
    print(len(train_data['Campaign_ID'].unique()))
    print(len(train_data['Account_ID'].unique()))
    print(len(train_data['Match_type_ID'].unique()))
    print(len(train_data['Device_ID'].unique()))


    # Data Exploration/Descriptive Analysis
    full_data=data_exploration_analysis(train_data,prediction_data)

    # Identify ID, Categorical, Numerical, Date variables if any
    cat_cols = ['Account_ID', 'Ad_group_ID', 'Campaign_ID', 'Device_ID', 'Keyword_ID', 'Match_type_ID']
    date_cols = ['Date']
    num_cols = ['Revenue', 'Clicks', 'Conversions']
    other_cols = ['Type']

    # Data Preprocessing
    full_data=data_preprocessing(full_data,cat_cols)

    # Coorelation matrix between diffrent independent (predictors) and dependent (target)
    print(str(full_data[full_data['Type'] == 'Train'].corr()))
    print(full_data.head(10))
    """
        Independent variables are highly uncorrelated, hence multicolinaerity problem will not exist. However, RPC values are also highly uncorrelated.
        Therefore, Ridge and Lasso regression are not intereting to this problem which are assumed to be good when independent varibles are highly correlated
    """

    # Identify target variable
    target_col=['RPC']

    # Splitting full_data into training and prediction data
    train=full_data[full_data['Type']=='Train']
    predict=full_data[full_data['Type']=='Predict']

    # Splitting training data into training (80%)and validation (20%) data
    train['is_train']=np.random.uniform(0,1,len(train))<=0.80
    Train, Validate =train[train['is_train']==True],train[train['is_train']==False]
    # Train, Validate =train[(train['is_train']==True) & (train['RPC']!=0.0)],train[(train['is_train']==False) & (train['RPC']!=0.0)]


    # Add 'is_train' column to other_cols
    other_cols=other_cols+['is_train']


    ### Choosing linear model
    # Considering all training data with category variables values As It Is
    model = forward_selected(Train[['Account_ID', 'Ad_group_ID', 'Campaign_ID', 'Device_ID', 'Keyword_ID', 'Match_type_ID', 'RPC']], 'RPC')
    print(str(model.model.formula)) # RPC ~ Device_ID + Match_type_ID + Account_ID + Campaign_ID + Ad_group_ID + Keyword_ID + 1
    print(str(model.rsquared_adj)) # 0.000385674891981
    #Low RSquared values indicate these independent variables do not define variance of dependent variable 'RPC' (target variable)

    # After converting category varibale (Device_ID, Match_type_ID) into dummy variables
    model = forward_selected(Train[['Account_ID', 'Ad_group_ID', 'Campaign_ID', 'Device_ID_0','Device_ID_1','Device_ID_2', 'Keyword_ID', 'Match_type_ID_0', 'Match_type_ID_1','Match_type_ID_2', 'RPC']], 'RPC')
    print(str( model.model.formula))  # RPC ~ Device_ID_2 + Match_type_ID_2 + Device_ID_0 + Account_ID + Campaign_ID + Ad_group_ID + Keyword_ID + Match_type_ID_1 + 1
    print(str(model.rsquared_adj))  # 0.00074716428297
    # RSquared values increases a bit, still far away from expected value


    ### Feature selection
    features=list(set(list(full_data.columns))-set(date_cols)-set(target_col)-set(other_cols)-set(num_cols))
    x_train=Train[list(features)].values
    y_train=Train['RPC'].values
    x_validate=Validate[list(features)].values
    y_validate=Validate['RPC'].values
    x_predict=predict[list(features)].values


    ### Regression Modelling and Evaluation
    # 1. Linear Regression (To confirm if varibles are linearly separated or best fit line is a straight line)
    mean_square_error,RSquare=linear_regression(x_train, y_train, x_validate, y_validate)
    print(str(mean_square_error))
    print(str(RSquare))
    # mean_squared_error: 519385.404547, RSquare score: 0.000386495457006
    # Predictions are not good due to lack of linearity relation between of varibles as expected

    # 2. Polynomial Regression: extending linear models with plynomial functions (To confirm if polynomial curve fits into the data points)
    mean_square_error,RSquare=polynomial_regression(x_train,y_train,x_validate,y_validate)
    print(str(mean_square_error))
    print(str(RSquare))
    # mean_squared_error: 510407.227463, RSquare score: 0.0012936464935
    # RSquare score on validation data increases compared to linear model but value is very less


    ### Since almost 98% of training data has Revenue 0, grouping data and find RPC as sum(revenue)/sum(clicks) for each group
    #grouping_features_list=['Account_ID', 'Ad_group_ID', 'Campaign_ID', 'Keyword_ID', 'Match_type_ID', 'Device_ID']
    #grouping_features_list=['Keyword_ID', 'Match_type_ID_0', 'Match_type_ID_1', 'Match_type_ID_2']
    grouping_features_list=['Keyword_ID', 'Match_type_ID_0','Match_type_ID_1','Match_type_ID_2','Device_ID_0','Device_ID_1','Device_ID_2']
    x_train,y_train=train_grouping(train,grouping_features_list)
    x_validate=Validate[grouping_features_list]
    y_validate=Validate['RPC']

    print(x_train.shape)
    print(y_train.shape)

    mean_square_error,RSquare,model=linear_regression(x_train, y_train, x_validate, y_validate)
    print(str(mean_square_error))
    print(str(RSquare))

    # mean_squared_error: 494212.379762, RSquare score: 0.000519865414236 when group by 'Account_ID', 'Ad_group_ID', 'Campaign_ID', 'Keyword_ID', 'Match_type_ID', 'Device_ID'
    # mean_squared_error: 626283.345572, RSquare score: 0.0000817006614914 when group by 'Account_ID', 'Ad_group_ID', 'Campaign_ID', 'Keyword_ID', 'Match_type_ID'
    # mean_squared_error: 554954.95308,  RSquare score: 0.000167463748957 when group by 'Ad_group_ID', 'Campaign_ID', 'Keyword_ID', 'Match_type_ID','Device_ID'
    # mean_squared_error: 463214.21261,  RSquare score: 0.0000790717316593 when group by 'Ad_group_ID', 'Campaign_ID', 'Keyword_ID', 'Match_type_ID'
    # mean_squared_error: 437754.496886, RSquare score: 0.000420399592246  when group by 'Keyword_ID', 'Match_type_ID_0', 'Match_type_ID_1','Match_type_ID_2','Device_ID_0','Device_ID_1','Device_ID_2'
    # mean_squared_error: 462735.213415, RSquare score: -0.00025548983471 when group by 'Keyword_ID', 'Match_type_ID'
    # mean_squared_error: 510258.029836, RSquare score: -0.000215123335169 when group by 'Keyword_ID', 'Match_type_ID_0', 'Match_type_ID_1', 'Match_type_ID_2'

    resultFile = open('bhansali_prediction.csv', 'w')
    print(model.predict(x_predict))
    print(model.predict(x_predict).shape)

    predictions=model.predict(x_predict)

    with open('prediction.p','wb') as fp:
        pickle.dump(predictions,fp)

    for label in predictions:
        resultFile.write(str(label[0]))
        resultFile.write('\n')


    mean_square_error,RSquare=polynomial_regression(x_train, y_train, x_validate, y_validate)
    print(str(mean_square_error))
    print(str(RSquare))
    # mean_squared_error: 655918.356669, score: 0.000653619243192 when group by 'Account_ID', 'Ad_group_ID', 'Campaign_ID', 'Keyword_ID', 'Match_type_ID', 'Device_ID'
    # mean_squared_error: 611018.408073, score: 0.000474923625903 when group by 'Ad_group_ID', 'Campaign_ID', 'Keyword_ID', 'Match_type_ID', 'Device_ID'
    # mean_squared_error: 443270.709312, score: 0.000318373637927 when group by 'Keyword_ID', 'Match_type_ID_0', 'Match_type_ID_1','Match_type_ID_2','Device_ID_0','Device_ID_1','Device_ID_2'

    # 3 RandomForest Regressor
    mean_square_error,RSquare=randomforest_regressor(x_train,y_train,x_validate,y_validate)
    print(str(mean_square_error))
    print(str(RSquare))
    # mean_squared_error: 444588.4558, R Square score: -0.00950279753028 without category variable into dummy variables
    # mean_squared_error: 473586.777762, R Square score: -0.0144142990942 when group by 'Keyword_ID', 'Match_type_ID_0', 'Match_type_ID_1','Match_type_ID_2','Device_ID_0','Device_ID_1','Device_ID_2'

    # 4. SVR
    # mean_square_error,RSquare = support_vector_regression(x_train,y_train,x_validate,y_validate)
    # print(str(mean_square_error))
    # print(str(RSquare))

if __name__ == '__main__':
    main()

'''
Next steps:
    1. We could over-sample our data. We could do some record engineering to create "fake" records that are very similar to those that generate revenue.
    This will increase the likelihood that our model catches on to what is driving revenue. I would have tried it if given more time.

    2. I tried to run SVM Rregressor on given data but it was taking huge amount to run. I tried to optimize it but couldn't complete the run on given data.

'''