# click-stream-mining
This project is my homework assignment for CSE 537 Artificial Intelligence, taught by Prof. I V Ramakrishnan
Problem statement:
Clickstream Mining with Decision Trees

The project is based on a task posed in KDD Cup 2000. It involves mining click-stream data collected from Gazelle.com, which sells legware products. Your task is to determine: Given a set of page views, will the visitor view another page on the site or will he leave?

The data set given to you is in a different form than the original. In particular it has discretized numerical values obtained by partitioning them into 5 intervals of equal frequency. This way, we get a data set where for each feature, we have a finite set of values. These values are mapped to integers, so that the data is easier for you to handle

Dataset: 5 files in .csv format

trainfeat.csv: Contains 40000 examples, each with 274 features in the form of a 40000 x 274 matrix.
trainlabs.csv: Contains the labels (class) for each training example (did the visitor view another page?)
testfeat.csv: Contains 25000 examples, each with 274 features in the form of a 25000 x 274 matrix.
testlabs.csv: Contains the labels (class) for each testing example.
featnames.csv: Contains the "names" of the features. These might useful if you need to know what the features represent.
The format of the files is not complicated, just rows of integers separated by empty spaces.

Stopping Criterion: Chi-squared criterion

What about split stopping? Suppose that the attribute that we want to split on is irrelevant. Then we would expect the positive and negative examples (labels 1 and 0) to be distributed according to a specific distribution. Suppose that splitting on attribute T, will produce sets {T_i} where i = 1 to m

Let p, n denote the number of positive and negative examples that we have in our dataset (not the whole set, the remaining one that we work on at this node). Let (N is the total number of examples in the current dataset):


be the expected number of positives and negatives in each partition, if the attribute is irrelevant to the class. Then the statistic of interest is:


where p_i, n_i are the positives and negatives in partition T_i. The main idea is that S is distributed (if the class is irrelevant) according to a chi-squared distribution with m-1 degrees of freedom.

Now we must compute the p-value. This is the probability of observing a value X at least as extreme as S coming from the distribution. To find that, we compute P(X >= S). The test is passed if the p-value is smaller than some threshold. Remember, the smallest that probability is, the more unlikely it is that S comes from a chi-squared distribution and consequently, that T is irrelevant to the class.

Instructions to run the code:
Required environment: Python 2.7 with scipy, pandas and copy installed.
command: python q1_classifier.py -p <pvalue> -f1 <train_dataset> -f2 <test_dataset> -o <output_file> -t <decision_tree>
Where:
<train_dataset> and <test_dataset> are .csv files
<output_file> is a csv file containing the predicted labels for the test dataset
<decision_tree> is your decision tree [a pkl file for storing decision tree]
<pvalue> is the p-value threshold as described above

Observations and conclusion:
1.	For p_value = 0.01
  No. of internal nodes: 26
  No. of leaf nodes: 105
Tree prediction accuracy:  0.75216
Output file prediction accuracy:  0.75216
Tree prediction matches output file
2.	For p_value = 0.05
  No. of internal nodes: 35
  No. of leaf nodes: 141
Tree prediction accuracy:  0.75324
Output file prediction accuracy:  0.75324
Tree prediction matches output file
3.	For p_value = 1.0
  No. of internal nodes: 265
  No. of leaf nodes:  1061
Tree prediction accuracy:  0.74832
Output file prediction accuracy:  0.74832
Tree prediction matches output file

As observed, number of nodes increases as p_value increases.
Standard decision trees have no learning bias and in such situation the training error may become zero. Nevertheless, this results into over-fitting. This over-fitting can be prevented if we build simpler tree by performing pruning. 
For p_value = 1.0, the whole tree is expanded and may result in over-fitting.
Whereas, in case of p_value = 0.01, excessive pruning takes place and it may result in under-fitting. 
Thus, we can conclude from that for p_value = 0.05, we get balanced tree and better accuracy.

