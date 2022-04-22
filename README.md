# Detection of Data Exfiltration via DNS

## Overview
Exfiltration of data over DNS and maintaining tunneled command and control communications for malware is one of the critical attacks exploited by cyber-attackers against enterprise networks to fetch valuable and sensitive data from their networks since DNS traffic is allowed to pass through firewalls by default, attackers can encode valuable information in DNS queries without fear of being detected.

## Solution
In this project, we introduce a real-time mechanism to detect exfiltration and tunneling of data over DNS through training a machine learning model that is capable of detecting anomalies in DNS queries.

## Random Forest

### Classification Report
<img src="https://github.com/alansary/Detection-of-Data-Exfiltration-via-DNS/blob/main/images/Random-Forest-Classification-Report.png" data-canonical-src="https://github.com/alansary/Detection-of-Data-Exfiltration-via-DNS/blob/main/images/Random-Forest-Classification-Report.png" width="600" height="300" alt="Classification Report" />

### Confusion Matrix
<img src="https://github.com/alansary/Detection-of-Data-Exfiltration-via-DNS/blob/main/images/Random-Forest-Confusion-Matrix.png" data-canonical-src="https://github.com/alansary/Detection-of-Data-Exfiltration-via-DNS/blob/main/images/Random-Forest-Confusion-Matrix.png" width="300" height="300" alt="Confusion Matrix" />

## Decision Tree

### Classification Report
<img src="https://github.com/alansary/Detection-of-Data-Exfiltration-via-DNS/blob/main/images/Decision-Tree-Classification-Report.png" data-canonical-src="https://github.com/alansary/Detection-of-Data-Exfiltration-via-DNS/blob/main/images/Decision-Tree-Classification-Report.png" width="600" height="300" alt="Classification Report" />

### Confusion Matrix
<img src="https://github.com/alansary/Detection-of-Data-Exfiltration-via-DNS/blob/main/images/Decision-Tree-Confusion-Matrix.png" data-canonical-src="https://github.com/alansary/Detection-of-Data-Exfiltration-via-DNS/blob/main/images/Decision-Tree-Confusion-Matrix.png" width="300" height="300" alt="Confusion Matrix" />

## XGBoost

### Classification Report
<img src="https://github.com/alansary/Detection-of-Data-Exfiltration-via-DNS/blob/main/images/XGBoost-Classification-Report.png" data-canonical-src="https://github.com/alansary/Detection-of-Data-Exfiltration-via-DNS/blob/main/images/XGBoost-Classification-Report.png" width="600" height="300" alt="Classification Report" />

### Confusion Matrix
<img src="https://github.com/alansary/Detection-of-Data-Exfiltration-via-DNS/blob/main/images/XGBoost-Confusion-Matrix.png" data-canonical-src="https://github.com/alansary/Detection-of-Data-Exfiltration-via-DNS/blob/main/images/XGBoost-Confusion-Matrix.png" width="300" height="300" alt="Confusion Matrix" />

## Experiments

### Model Evaluation
The three models have the same accuracy and f1-score for both classes. A reasonable way to select the champion model is based on the number and cost of false negatives and false positives. Our solution should put more weight on false negatives as the cost of classifying a malicious DNS query as benign is more than the cost of classifying a benign DNS query as malicious and therefore we will select the model that has reasonable false negatives and false positives. Based on this criteria our champion model is XGBoost.

### Hyperparameter Tuning
We searched for the best hyperparameters using randomized search and obtained a little higher accuracy with the new parameters but also false negatives increased to 53 instead of 25 and so we used the default hyperparameter values instead.

<img src="https://github.com/alansary/Detection-of-Data-Exfiltration-via-DNS/blob/main/images/Hyperparameter-Tuning.png" data-canonical-src="https://github.com/alansary/Detection-of-Data-Exfiltration-via-DNS/blob/main/images/Hyperparameter-Tuning.png" width="800" height="150" alt="Hyperparameter Tuning" />

### Results
The highest accuracy we could get is 82%, we can increase this accuracy using stateful features in addition to our stateless features. As we can see DNS exfiltration detection is achievable easily using machine learning algorithms having the advantage of real-time fast detection but as everything comes with a cost, we “must” tune our model regularly to keep its accuracy and evaluation metrics high and to enhance the model further.
