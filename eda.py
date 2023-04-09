import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def exploratory_data_analysis(comments_df):
    
    print(comments_df.shape)
    print(comments_df.head())
    print(comments_df.info)
    print(comments_df.describe())
    print(comments_df.nunique())
    print(comments_df.isnull().sum())

    # Plot for representing the distribution for length of words
    comment_length = comments_df['comment_text'].str.split().apply(len)
    sns.histplot(comment_length, bins=50)
    plt.title("Distribution for Length of words")
    plt.xlabel("Number of words")
    plt.ylabel("Density")
    plt.xlim(0, 800)
    plt.show()

    rowSums = comments_df.iloc[:, 2:].sum(axis=1)
    clean_comments_count = (rowSums == 0).sum(axis=0)
    print("Total number of comments = ", len(comments_df))
    print("Number of clean comments = ", clean_comments_count)
    print("Number of comments with labels =", (len(comments_df) - clean_comments_count))

    categories = list(comments_df.columns.values)
    categories = categories[2:]
    print(categories)

    counts = []
    for category in categories:
        counts.append((category, comments_df[category].sum()))
    counts.append(('clean', clean_comments_count))
    df_stats = pd.DataFrame(counts, columns=['category', 'number of comments'])
    print(df_stats)

    # Plot showing comments under each category
    sns.set(font_scale=1.5)
    plt.figure(figsize=(15, 8))
    fig = sns.barplot(x=categories, y=comments_df.iloc[:, 2:].sum().values)
    plt.title("Comments in each category", fontsize=25)
    plt.ylabel('Number of comments', fontsize=16)
    plt.xlabel('Comment Type ', fontsize=16)
    # adding counts for the labels
    rects = fig.patches
    labels = comments_df.iloc[:, 2:].sum().values
    for rect, label in zip(rects, labels):
        height = rect.get_height()
        fig.text(rect.get_x() + rect.get_width() / 2, height + 4, label, ha='center', va='bottom', fontsize=16)
    plt.show()

    print("Here, the 'clean' comment category has been disregarded as the count for the same is very huge as compared" +
          "to other categories [see above].Hence, to remove the skewness and to clearly show the low-count(s) of hatred" +
          "categories 'clean' category is skipped.")

    rowSums = comments_df.iloc[:, 2:].sum(axis=1)
    multiLabel_count = rowSums.value_counts()
    multiLabel_count = multiLabel_count.iloc[1:]

    # Plot showing comments under multiple categories
    sns.set(font_scale=1.5)
    plt.figure(figsize=(15, 8))
    fig = sns.barplot(x=multiLabel_count.index, y=multiLabel_count.values)
    plt.title("Comments having multiple categories ")
    plt.ylabel('Number of comments', fontsize=16)
    plt.xlabel('Number of categories', fontsize=16)
    # adding counts for the labels
    rects = fig.patches
    labels = multiLabel_count.values
    for rect, label in zip(rects, labels):
        height = rect.get_height()
        fig.text(rect.get_x() + rect.get_width() / 2, height + 4, label, ha='center', va='bottom', fontsize='16')
    plt.show()

    correlation_val = comments_df.corr()
    print(correlation_val)

    # For better visualization, we will represent these values via Heatmap.
    plt.figure(figsize=(11, 8.5))
    sns.heatmap(comments_df.corr(), annot=True)
    plt.suptitle('Heatmap of Categories available', size=16)
    plt.xlabel("Categories")
    plt.ylabel("Categories")
    plt.show()

    # Plot showing Toxic comments vs Clean Comments count
    toxic_comments = comments_df[comments_df[categories].sum(axis=1) > 0]
    clean_comments = comments_df[comments_df[categories].sum(axis=1) == 0]
    pd.DataFrame(dict(toxic=[len(toxic_comments)], clean=[len(clean_comments)])).plot(kind='barh')

