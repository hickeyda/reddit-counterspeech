# Datasheet for the counterspeech dataset used in "Hostile Counterspeech Drives Users From Hate Subreddits"

Questions from the [Datasheets for Datasets](https://arxiv.org/abs/1803.09010) paper, v7.

Jump to section:

- [Motivation](#motivation)
- [Composition](#composition)
- [Collection process](#collection-process)
- [Preprocessing/cleaning/labeling](#preprocessingcleaninglabeling)
- [Uses](#uses)
- [Distribution](#distribution)
- [Maintenance](#maintenance)

## Motivation

_The questions in this section are primarily intended to encourage dataset creators
to clearly articulate their reasons for creating the dataset and to promote transparency
about funding interests._

### For what purpose was the dataset created? 

This dataset was created to train a model for counterspeech detection in hate subreddits. The model was then applied to assess how effective counterspeech is for reducing engagement in hate subreddits.

### Who created the dataset (e.g., which team, research group) and on behalf of which entity (e.g., company, institution, organization)?
Omitted to preserve anonymity.

### Who funded the creation of the dataset? 

Omitted to preserve anonymity.

### Any other comments?

## Composition

_Most of these questions are intended to provide dataset consumers with the
information they need to make informed decisions about using the dataset for
specific tasks. The answers to some of these questions reveal information
about compliance with the EU’s General Data Protection Regulation (GDPR) or
comparable regulations in other jurisdictions._

### What do the instances that comprise the dataset represent (e.g., documents, photos, people, countries)?

Each instance represents a series of three Reddit comments: context, newcomer, and reply. The context is the parent post/comment of the newcomer, while the reply is a reply to the newcomer. Both the reply and newcomer comments are labeled for whether they are counterspeech, and the reply comment is labeled for whether or not it is a personal attack.

### How many instances are there in total (of each type, if appropriate)?

There are 450 triplets of context, newcomer, and reply comments.

### Does the dataset contain all possible instances or is it a sample (not necessarily random) of instances from a larger set?

The dataset is a sample of comments from a set of 25 hate subreddits.

### What data does each instance consist of? 

The data consist of raw text from Reddit comments.

### Is there a label or target associated with each instance?

Yes, the reply comments are labeled for whether or not they constitute counterspeech and/or personal attacks and the newcomer comments are labeled for whether or not they constitute counterspeech.

### Is any information missing from individual instances?

Yes, some of the context comments/submissions are missing as they could not be obtained from the Pushshift API.

### Are relationships between individual instances made explicit (e.g., users’ movie ratings, social network links)?

Individual instances comprise relationships among users, though individual instances are mostly unrelated.

### Are there recommended data splits (e.g., training, development/validation, testing)?

We used a training/validation/testing split of 70/15/15 to train the counterspeech detection model in order to have sufficient data to estimate the performance of the model. However, as the dataset is small, we recommend splitting the data using various random seeds to estimate the model performance. 

### Are there any errors, sources of noise, or redundancies in the dataset?

Some of the context comments are missing as they could not be obtained from Pushshift. Additionally, as the dataset was annotated by several annotators, there are some examples that annotators disagreed on more than others, which could be a potential source of noise.

### Is the dataset self-contained, or does it link to or otherwise rely on external resources (e.g., websites, tweets, other datasets)?

The dataset is self-contained.

### Does the dataset contain data that might be considered confidential (e.g., data that is protected by legal privilege or by doctor-patient confidentiality, data that includes the content of individuals’ non-public communications)?

No.

### Does the dataset contain data that, if viewed directly, might be offensive, insulting, threatening, or might otherwise cause anxiety?

Yes, the dataset contains several examples of hate speech directed at women, various racial/ethnic groups, various nationalities, and members of the LGBTQ+ community.

### Does the dataset relate to people? 

Yes.

### Does the dataset identify any subpopulations (e.g., by age, gender)?

No.

### Is it possible to identify individuals (i.e., one or more natural persons), either directly or indirectly (i.e., in combination with other data) from the dataset?

Using Pushshift, these comments could be traced back to individual's Reddit usernames, though this would not necessarily identify them if they do not reveal their personal information publicly on Reddit.

### Does the dataset contain data that might be considered sensitive in any way (e.g., data that reveals racial or ethnic origins, sexual orientations, religious beliefs, political opinions or union memberships, or locations; financial or health data; biometric or genetic data; forms of government identification, such as social security numbers; criminal history)?

No.

### Any other comments?

## Collection process

_\[T\]he answers to questions here may provide information that allow others to
reconstruct the dataset without access to it._

### How was the data associated with each instance acquired?

The Reddit comments were obtained from Pushshift (Baumgartner et al.). To determine which comments were counterspeech or personal attacks, a team of annotators labeled each comment.

### What mechanisms or procedures were used to collect the data (e.g., hardware apparatus or sensor, manual human curation, software program, software API)?

Pushshift comments were accessed from the Pushshift API. The labels provided by human annotators were validated using the Fleiss Kappa metric.

### If the dataset is a sample from a larger set, what was the sampling strategy (e.g., deterministic, probabilistic with specific sampling probabilities)?

The data are from a stratified random sample, with an equal number of comments being randomly sampled from each subreddit. Within each subreddit, a counterspeech detection model trained on a dataset from a different domain was used to predict likely counterspeech, and an equal number of newcomer/reply pairs were sampled from each possible classification (newcomer counterspeech/reply counterspeech, newcomer counterspeech/reply not counterspeech, etc.)

### Who was involved in the data collection process (e.g., students, crowdworkers, contractors) and how were they compensated (e.g., how much were crowdworkers paid)?

Undergraduate research assistants were involved in the labeling of data.

### Over what timeframe was the data collected?

The data were collected between May and August of 2023. However, the creation of the Reddit posts range from 2013 to 2021.

### Were any ethical review processes conducted (e.g., by an institutional review board)?

No.

### Does the dataset relate to people?

Yes.

### Did you collect the data from the individuals in question directly, or obtain it via third parties or other sources (e.g., websites)?

The data were collected via a third party, namely the Pushshift API (Baumgartner et al.).

### Were the individuals in question notified about the data collection?

No.

### Did the individuals in question consent to the collection and use of their data?

The data obtained are from public Reddit posts, though the individuals in question did not consent to their data being used for this specific purpose.

### If consent was obtained, were the consenting individuals provided with a mechanism to revoke their consent in the future or for certain uses?

### Has an analysis of the potential impact of the dataset and its use on data subjects (e.g., a data protection impact analysis) been conducted?

No.

### Any other comments?

## Preprocessing/cleaning/labeling

_The questions in this section are intended to provide dataset consumers with the information
they need to determine whether the “raw” data has been processed in ways that are compatible
with their chosen tasks. For example, text that has been converted into a “bag-of-words” is
not suitable for tasks involving word order._

### Was any preprocessing/cleaning/labeling of the data done (e.g., discretization or bucketing, tokenization, part-of-speech tagging, SIFT feature extraction, removal of instances, processing of missing values)?

The data were assigned labels based on whether or not they constituted counterspeech or attacks.

### Was the “raw” data saved in addition to the preprocessed/cleaned/labeled data (e.g., to support unanticipated future uses)?

The raw text of the comments are available in the same file as the labels.

### Is the software used to preprocess/clean/label the instances available?

Labels were manually assigned to the dataset.

### Any other comments?

## Uses

_These questions are intended to encourage dataset creators to reflect on the tasks
for which the dataset should and should not be used. By explicitly highlighting these tasks,
dataset creators can help dataset consumers to make informed decisions, thereby avoiding
potential risks or harms._

### Has the dataset been used for any tasks already?

Yes, the dataset has been used to train a counterspeech detection model.

### Is there a repository that links to any or all papers or systems that use the dataset?

No.

### What (other) tasks could the dataset be used for?

The dataset could be used to train a model to detect personal attacks, or could be used to supplement other datasets containing examples of counterspeech/attacks to train machine learning models.

### Is there anything about the composition of the dataset or the way it was collected and preprocessed/cleaned/labeled that might impact future uses?

No.

### Are there tasks for which the dataset should not be used?

The dataset should not be used to train text-generation models, as it contains many instances of offensive speech, which could lead to the generation of text that is offensive.

### Any other comments?

## Distribution

### Will the dataset be distributed to third parties outside of the entity (e.g., company, institution, organization) on behalf of which the dataset was created? 

Yes, the dataset will be distributed to anyone who wishes to use it.

### How will the dataset will be distributed (e.g., tarball on website, API, GitHub)?

The dataset will be publicly available on GitHub.

### When will the dataset be distributed?

The dataset will be distributed upon publication of the paper that accompanies it.

### Will the dataset be distributed under a copyright or other intellectual property (IP) license, and/or under applicable terms of use (ToU)?

No.

### Have any third parties imposed IP-based or other restrictions on the data associated with the instances?

No.

### Do any export controls or other regulatory restrictions apply to the dataset or to individual instances?

No.

### Any other comments?

## Maintenance

_These questions are intended to encourage dataset creators to plan for dataset maintenance
and communicate this plan with dataset consumers._

### Who is supporting/hosting/maintaining the dataset?

Omitted to preserve anonymity.

### How can the owner/curator/manager of the dataset be contacted (e.g., email address)?

Omitted to preserve anonymity.

### Is there an erratum?

No.

### Will the dataset be updated (e.g., to correct labeling errors, add new instances, delete instances)?

If any errors are discovered, the dataset will be updated accordingly.

### If the dataset relates to people, are there applicable limits on the retention of the data associated with the instances (e.g., were individuals in question told that their data would be retained for a fixed period of time and then deleted)?

No.

### Will older versions of the dataset continue to be supported/hosted/maintained?

Yes, older versions of the dataset will be available on GitHub.

### If others want to extend/augment/build on/contribute to the dataset, is there a mechanism for them to do so?

If others wish to build on the dataset and host it on the GitHub the dataset is hosted on, they may contact the authors and ask them to do so. Alternatively, they can host it on their own GitHub account.

### Any other comments?
