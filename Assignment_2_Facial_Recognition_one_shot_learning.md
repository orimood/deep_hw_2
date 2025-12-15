**Ben-Gurion University of the Negev**

**Faculty of Engineering Sciences**

**Department of Software and Information systems**** ****Engineering**

# **Deep Learning**

# **Assignment 2**

### <u>**The purpose of the assignment**</u>

Enabling students to experiment with building a convolutional neural net and using it on a real-world dataset  and problem. In addition to practical knowledge in the “how to” of building the network, an additional goal is the integration of useful logging tools for gaining better insight into the training process. Finally, the students are expected to read, understand and (loosely) implement a scientific paper.

### <u>**Submission instructions:**</u>

The assignment due date: 24/12/2024

The assignment is to be carried out using PyTorch.

Submission in <u>**pairs**</u> only. Only <u>**one copy**</u> of the assignment needs to be uploaded to Moodle.

Plagiarism of any kind (e.g., GitHub) is forbidden.

The entire project will be submitted as a single zip file, containing both the report (in PDF form only) and the code. It is the students’ responsibility to make sure that the file is valid (i.e. can be opened correctly).

The report accompanying the code needs to be a .**docx file **(use Calibri font, text size 12, margins of 2.5cm). The report itself will not exceed **6 pages **of text. Figures (graphs, tables, etc) can be placed an appendix, and must be referenced in the main body of report.

## <u>**Introduction**</u>

In this assignment, you will use convolutional neural networks (CNNs) to carry out the task of facial recognition. As shown in class, CNNs are the standard approach for analyzing image-based datasets. More specifically, you will implement a one-shot classification solution. Wikipedia defines one-shot learning as follows:

> *“…** an object categorization problem, found mostly in computer vision. Whereas most machine learning based object categorization algorithms require training on hundreds or thousands of samples/images and very large datasets, one-shot learning aims to learn information about object categories from one, or only a few, training samples/images.**”*

Your work will be based on the paper  . Your goal, like that of the paper, is to successfully execute a one-shot learning task for previously unseen objects. Given two facial images of previously unseen persons, your architecture will have to successfully determine whether they are the same person. While we encourage you to use the architecture described in this paper as a starting point, you are more than welcome to explore other possibilities.

## <u>**Instructions**</u>

Read the above-mentioned paper.

Use the following dataset -

Download the dataset. Note: there are several versions of this dataset, use the version found  (it’s called LFW-a, and is also used in the DeepFace paper).

Use the following train and test sets to train your model:  \\ . **[Remember - you will use your test set to perform one-shot learning****.**** ****T****his division is ****set up**** so**** that**** no subject from test set ****is**** included in the train**** set****]**. Please note it is often a recommended to use a validation set when training your model. Make your own decision whether to use one and what percentage of (training) samples to allocate.

In your report, include an analysis of the dataset (size, number of examples – in total and per class – for the train and test sets, etc). Also provide the full experimental setup you used – batch sizes, the various parameters of your architecture, stopping criteria and any other relevant information. A good rule of thumb: if asked to recreate your work, a person should be able to do so based on the information you provide in your report.

Implement a <u>Siamese network</u> architecture while using the above-mentioned paper as a reference.

Provide a complete description of your architecture: number of layers, dimensions, filters etc. Make sure to mention parameters such as learning rates, optimization and regularization methods, and the use (if exists) of batchnorm.

Explain the reasoning behind the choices made in answer to the previous section. If your choices were the result of trial and error, please state the fact and describe the changes made throughout your experiments. Choosing certain parameter combinations because they appeared in a previously published paper is a perfectly valid reason.

In addition to the details requested above, your report needs to include an analysis of your architecture’s performance. Please include the following information:

Convergence times, final loss and accuracy on the test set and holdout set

Graphs describing the loss on the training set throughout the training process

Performance when experimenting with the various parameters

Please include examples of accurate and misclassifications and try to determine why your model was not successful.

Any other information you consider relevant or found useful while training the model

Please note the that report needs to reflect your decision-making process throughout the assignment. Please include all relevant information.

Please note that your work will not be evaluated solely on performance, but also on additional elements such as code correctness and documentation, a complete and clear documentation of your experimental process, analysis of your results and breadth and originality (where applicable).

*Figure 1 - Siamese network for facial recognition*
