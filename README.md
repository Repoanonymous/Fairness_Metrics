# Fairness Metrics: All you need to know

Testing machine learning software for ethical bias has become a pressing current concern. Recent research has proposed a plethora of  fairness metrics including the 30 explored in this paper.  How can any fairness tool satisfy such a diverse range of goals?

While we cannot completely simplify the task of fairness testing, we can certainly reduce the problem. This paper shows that many of those supposedly different fairness metrics effectively  measure the same thing. Based on experiments using 7 real-world datasets, we find that 26 classification metrics can be clustered into 7 and 4 dataset metrics can be clustered into 3 clusters. Further, that reduced set actually predict for different things. Hence, it is no longer necessary (or even possible) to satisfy all these fairness metrics. 

In summary, to dramatically simplify the fairness testing problem, we recommend (a) determining what type of fairness is desirable (and we offer a handful of such types); then (b) look up those types in our clusters; then (c) just test for one item per cluster.

# Code Description

1. The Data folder contains the data use in this paper.
2. The results folder contains already generated results.
3. The src folder contains the code for generating all the RQs and initial results.

# Tutorial

1. To run create the results for analysis run initial_run.sh file. This will use 7 datasets on 3 models used in this study to generate Baseline, Reweighing and Meta Fair Classifier models. This task requires around 24 hours to complete.
2. To generate the results for RQ1 Table 5 and Table 6, run RQ1.sh
3. To generate the clusters as described in RQ2 Table 5 and Table 6, run RQ2.sh
4. To generate the results of RQ3 Table 8, run RQ3.sh
5. To generate the sensitivity results of RQ4 Table 7, run RQ4.sh
