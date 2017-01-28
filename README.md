# Understanding Customer Conversion with Snowplow Web Event Tracking
I apply machine learning (ML) techniques to Snowplow web event data to understand how variation in marketing site experiences might correlate to customer conversion.

### Data Transformation 
The distillation of the raw data into a transformed feature set with labels is handled by the iPython notebook 'Notebook 1 - Data Munging.'
<div align="center">
<img src="https://github.com/b-knight/Understanding-Customer-Conversion-with-Snowplow-Web-Event-Tracking/blob/master/Images/Data_Transformation.png" width="500" height="400" />


</div>
### Exploratory Analysis
The exploratory analysis of the initial feature set is handled in the iPython notebook 'Notebook 2 - Exploratory Analysis.'
<div align="center">
<img src="https://github.com/b-knight/Understanding-Customer-Conversion-with-Snowplow-Web-Event-Tracking/blob/master/Images/exploratory_analysis-labels.png" width="500" height="400" />
</div>

<div align="center">
<p align="center"><b>Summary Statistics: Means and Standard Deviations of Spare Feature Space</b></p>
<img src="https://github.com/b-knight/Understanding-Customer-Conversion-with-Snowplow-Web-Event-Tracking/blob/master/Images/exploratory_analysis-feature_means.png" align="middle" width="420" height="320" />
<img src="https://github.com/b-knight/Understanding-Customer-Conversion-with-Snowplow-Web-Event-Tracking/blob/master/Images/exploratory_analysis-feature_sds.png" align="middle" width="420" height="320" />
</div>

<div align="center">
<p align="center"><b>Results: SVM with RBF Kernel and Linear SVM Default Settings</b></p>
<img src="https://github.com/b-knight/Understanding-Customer-Conversion-with-Snowplow-Web-Event-Tracking/blob/master/Images/non-optimized_rbf_kernel.png" width="420" height="320" />
<img src="https://github.com/b-knight/Understanding-Customer-Conversion-with-Snowplow-Web-Event-Tracking/blob/master/Images/non-optimized_linear_svm.png" width="420" height="320" />
</div>
