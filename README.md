# Fairness Metrics: All you need to know

Testing machine learning software for ethical bias has become a pressing current concern. Recent research has proposed a plethora of  fairness metrics including the 30 explored in this paper.  How can any fairness tool satisfy such a diverse range of goals?

While we cannot completely simplify the task of fairness testing, we can certainly reduce the problem. This paper shows that many of those supposedly different fairness metrics effectively  measure the same thing. Based on experiments using 7 real-world datasets, we find that 26 classification metrics can be clustered into 7 and 4 dataset metrics can be clustered into 3 clusters. Further, that reduced set actually predict for different things. Hence, it is no longer necessary (or even possible) to satisfy all these fairness metrics. 

In summary, to dramatically simplify the fairness testing problem, we recommend (a) determining what type of fairness is desirable (and we offer a handful of such types); then (b) look up those types in our clusters; then (c) just test for one item per cluster.

