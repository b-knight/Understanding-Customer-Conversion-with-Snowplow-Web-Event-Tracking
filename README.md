# Understanding Customer Conversion with Snowplow Web Event Tracking <br> <sub> Benjamin S. Knight, January 27th 2017 </sub>

### Project Overview
Here I apply machine learning techniques to Snowplow web event data to infer whether trial account holders will become paying customers based on their history of visiting the marketing site. By predicting which trial account holders have the greatest likelihood of adding a credit card and converting to paying customers, we can more efficiently deploy scarce Sales Department resources. 

[Snowplow](http://snowplowanalytics.com/) is a web event tracker capable of handling tens of millions of events per day. The Snowplow data contains far more detail than the [MSNBC.com Anonymous Web Data Set](https://archive.ics.uci.edu/ml/datasets/MSNBC.com+Anonymous+Web+Data) hosted by the University of California, Irvine’s Machine Learning Repository. At the same time, we do not have access to demographic data as was the case with the [Event Recommendation Engine Challenge](https://www.kaggle.com/c/event-recommendation-engine-challenge) hosted by [Kaggle](https://www.kaggle.com/). Given the origin of the data, there is no industry-standard benchmark for model performance. Rather, assessing baseline feasibility is a key objective of this project. 

### Problem Statement
To what extent can we infer a visitor’s likelihood of becoming a paying customer based upon that visitor’s activity history on the company marketing site? We are essentially confronted with a binary classification problem. Will the trial account in question add a credit card (cc date added IS NOT NULL ‘yes’/‘no’)? This labeling information is contained in the ‘cc’ column within the file ‘munged df.csv.’

### Snowplow Web Event Data
The raw Snowplow data available is approximately 15 gigabytes spanning over 300 variables and tens of millions of events from November 2015 to January 2017. When we omit fields that are not in active use, are redundant, contain personal identifiable information (P.I.I.), or which cannot have any conceivable baring on customer conversion, then we are left with 14.6 million events spread across 22 variables.

|<sub>Snowplow Variable Name         |<sub>Snowplow Variable Description                                                  |
| ---------------------------------- |-----------------------------------------------------------------------------------------| 
| <sub>*event_id*            | <sub>The unique Snowplow event identifier                                                 |
| <sub>*account_id*          | <sub>The account number if an account is associated with the domain userid                |
| <sub>*reg_date*            | <sub>The date an account was registered                                                   |
| <sub>*cc_date_added*       | <sub>The date a credit card was added                                                     |
| <sub>*collector_tstamp*    | <sub>The timestamp (in UTC) when the Snowplow collector first recorded the event          |
| <sub>*domain_userid* | <sub>This corresponds to a Snowplow cookie and will tend to correspond to a single internet device|
| <sub>*domain_sessionidx*   | <sub>The number of sessions to date that the domain userid has been tracked               |
| <sub>*domain_sessionid*    | <sub>The unique identifier for the Snowplow cookie/session                                |
| <sub>*event_name*          | <sub>The type of event recorded                                                           |
| <sub>*geo_country*         | <sub>The ISO 3166-1 code for the country that the visitor’s IP address is located         |
| <sub>*geo_region_name*     | <sub>The ISO-3166-2 code for country region that the visitor’s IP address is in           |
| <sub>*geo_city*            | <sub>The city the visitor’s IP address is in                                              |
| <sub>*page_url*            | <sub>The page URL                                                                         |
| <sub>*page_referrer*       | <sub>The URL of the referrer (previous page)                                              |
| <sub>*mkt_medium*          | <sub>The type of traffic source (e.g. ’cpc’, ’affiliate’, ’organic’, ’social’)            |
| <sub>*mkt_source*          | <sub>The company / website where the traffic came from (e.g. ’Google’, ’Facebook’)        |
| <sub>*se_category*         | <sub>The event type                                                                       |
| <sub>*se_action*           | <sub>The action performed / event name (e.g. ’add-to-basket’, ’play-video’)               |
| <sub>*br_name*             | <sub>The name of the visitor’s browser                                                    |
| <sub>*os_name*             | <sub>The name of the vistor’s operating system                                            |
| <sub>*os_timezone*         | <sub>The client’s operating system timezone                                               |
| <sub>*dvce_ismobile*       | <sub>Is the device mobile? (1 = ’yes’)                                                    |

### Data Transformation 
I use the phrase 'variable' as opposed to 'feature', since this dataset will need to undergo substantial transformation before we can employ any supervised learning technique. Each row has an 'event_id' along with an 'event_name' and a ‘page url.’ The event id is the row’s unique identifier, the event name is the type of event, and the page url is the URL within the marketing site where the event took place.

The distillation of the raw data into a transformed feature set with labels is handled by the iPython notebook 'Notebook 1 - Data Munging.' In transforming the data, we will need to create features by creating combinations of event types and distinct URLs, and counting the number of occurrences while grouping on accounts. For instance, if ‘.../pay-ment plan.com’ is a frequent page url, then the number of page views on payment plan.com would be one feature, the number of page pings would be another, as would the number of web forms submitted, and so forth. Given that there are six distinct event types and dozens of URLs within the marketing site, then the feature space quickly expands to encompass hundreds of features. This feature space will only widen as we add additional variables to the mix including geo region, number of visitors per account, and so forth.

<div align="center">
<img src="https://github.com/b-knight/Understanding-Customer-Conversion-with-Snowplow-Web-Event-Tracking/blob/master/Images/Data_Transformation.png" align="middle" width="500" height="400" />
</div>
<br>
With the raw data transformed, our observations are no longer individual events but indivual accounts spanning the period November 2015 to January 2017. Our data set has 16,607 accounts and 581 features. The probability of an account sucessfully converting to a paying customer is approximately 6%.

### Exploratory Analysis
The exploratory analysis of the initial feature set is handled in the iPython notebook 'Notebook 2 - Exploratory Analysis.' Again, or data is highly unbalanced. Only 6% of the binary labels contained in the column 'cc' are of the class "paying customer." Another feature of the data is how sparse it is, as the following histograms of feature means and standard deviations make clear.

<div align="center">
<img src="https://github.com/b-knight/Understanding-Customer-Conversion-with-Snowplow-Web-Event-Tracking/blob/master/Images/exploratory_analysis-labels.png" align="middle" width="500" height="400" />
</div>

<div align="center">
<p align="center"><b>Summary Statistics: Means and Standard Deviations of Spare Feature Space</b></p>
<img src="https://github.com/b-knight/Understanding-Customer-Conversion-with-Snowplow-Web-Event-Tracking/blob/master/Images/exploratory_analysis-feature_means.png" align="middle" width="420" height="320" />
<img src="https://github.com/b-knight/Understanding-Customer-Conversion-with-Snowplow-Web-Event-Tracking/blob/master/Images/exploratory_analysis-feature_sds.png" align="middle" width="420" height="320" />
</div>

### Establishing a Baseline 
<p> How do we know if our ultimate model is any good? To establish a baseline, I implement a Support Vector Machine (SVM) model and a linear SVM model, both with their default settings. These initial models were created using a 90% / 10% split of training data (14,946 accounts) versus testing datas (1,661 accounts).

</p>
<div align="center">
<p align="center"><b>Results: SVM with RBF Kernel and Linear SVM (Default Hyper-Parameter Settings)</b></p>
<img src="https://github.com/b-knight/Understanding-Customer-Conversion-with-Snowplow-Web-Event-Tracking/blob/master/Images/non-optimized_rbf_kernel.png" width="420" height="320" />
<img src="https://github.com/b-knight/Understanding-Customer-Conversion-with-Snowplow-Web-Event-Tracking/blob/master/Images/non-optimized_linear_svm.png" width="420" height="320" />
</div>

### Bayesian Optimization 
In theory, we should be able to improve upon the baseline models by tuning the models' hyper-parameters. Our primary hyper-parameters of interest are 'C'  and 'gamma' for the SVM + RBF model, and just 'C' for the linear SVM model. Recall that C is the penalty parameter - how much we penalize our model for incorrect classifications. A higher C value holds the promise of greater accuracy, but at the risk of overfitting. The selection of the the gamma hyper-parameter determines the variance of the distributions generated by the [RBF kernel](https://www.youtube.com/watch?v=3liCbRZPrZA), with a large gamma tending to lead to higher bias but lower variance. 

The C and gamma hyper-parameters can vary by several orders of magnitude, so finding the optimal configuration is no small task. Employing a [grid search](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) can be computationally expensive - almost prohibitatively expensive without parallel computing resources. Fortunately, we do not have to exhaustively  

<div align="center">
<p align="center"><b>An Acquisition Function Combing a Unidimensional Space for Three Iterations</b></p>
<img src="https://github.com/b-knight/Understanding-Customer-Conversion-with-Snowplow-Web-Event-Tracking/blob/master/Images/bayesian_optimization.png" align="middle" width="585" height="720" />
</div>

### Results 
At first glance, it would appear that the Bayesian optimization worsened the predictive power of the RBF and linear SVM models. Whereas the default RBF model yielded an AUC of 0.26, 
<div align="center">
<p align="center"><b>Results: SVM with RBF Kernel and Linear SVM (Default Hyper-Parameter Settings)</b></p>
<img src="https://github.com/b-knight/Understanding-Customer-Conversion-with-Snowplow-Web-Event-Tracking/blob/master/Images/optimized_SVM_with_RBF.png" width="430" height="430" />
<img src="https://github.com/b-knight/Understanding-Customer-Conversion-with-Snowplow-Web-Event-Tracking/blob/master/Images/optimized_Linear_SVM.png" width="430" height="430" />
</div>

<div align="center">
<p align="center"><b>Results: Comparision of F2 Scores </b></p>
</div>

|              Model Used                 | With Default Settings | With Hyper-Parameter Tuning |
| :-------------------------------------- | :-------------------: | :--------------------------:| 
| K-Nearest Neighbors (Baseline)          |         0.25          |               -             |
| Support Vector Machines with RBF Kernel |         0.0           |                             |
| Linear Support Vector Machines          |                       |                             |
   

### Possible Next Steps
* Estimating variance
* Possible redefinition of success metric to weigh recall more highly than precision 
* Dimensionality reduction via [LDA](https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation)
* Use of [Random Forests](https://en.wikipedia.org/wiki/Random_forest) 
* Use of [OneClassSVM](http://scikit-learn.org/stable/modules/generated/sklearn.svm.OneClassSVM.html#sklearn.svm.OneClassSVM)
* Accounting for sparse feature space
