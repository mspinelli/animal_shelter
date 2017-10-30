import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_feature(feature, data, filter_=None, legend_offset=-0.3, width_inches=16.5,
                 tick_font_size=14, label_font_size=14, title=None):
    f, axs = plt.subplots(ncols=2)
    f.set_size_inches(width_inches, 5)
    axs[0].xaxis.label.set_size(label_font_size)
    axs[1].xaxis.label.set_size(label_font_size)
    if title:
        axs[0].set_title(title, fontsize=label_font_size)
        axs[1].set_title(title, fontsize=label_font_size)

    if type(filter_) != pd.Series:
        filter_ = np.full(len(data), True, dtype=bool)

    (data[filter_]
     .groupby(feature)
     .OutcomeType
     .value_counts(normalize=True)
     .unstack()
     .plot(fontsize=tick_font_size, kind="bar", stacked=True, ax=axs[0],
           edgecolor='k', linewidth=0.5, width=1, legend=False))

    (data[filter_]
     .groupby(feature)
     .OutcomeType.value_counts()
     .unstack()
     .plot(fontsize=tick_font_size, kind="bar", stacked=True, ax=axs[1],
           edgecolor='k', linewidth=0.5, width=1)
     .legend(fontsize=label_font_size, ncol=5,
             bbox_to_anchor=(-.1, legend_offset), loc='center'))

def visualize_folds(data, train, test):
    from IPython.core import display as ICD
    print 'Number of training samples: {}\n'.format(len(train[0]))
    print 'Training sample size per fold:'
#     summary = pd.DataFrame([
#         data.iloc[f,:].groupby('Ordinal_Outcome')
#         .count()
#         .ID.rename('Fold {}'.format(i+1))
#         for i, f in enumerate(train)])
    summary = pd.DataFrame([data.iloc[train[i]].Ordinal_Outcome.value_counts(sort=False)
                            .rename('Fold {}'.format(0+1))
                            for i, indices in enumerate(train)])
    summary['Total'] = summary.sum(axis=1)
    # summary.columns = summary.columns.rename('')
    ICD.display(summary)

    print('Validation samples per fold:')
    summary = pd.DataFrame([data.iloc[test[i]].Ordinal_Outcome.value_counts(sort=False)
                            .rename('Fold {}'.format(0+1))
                            for i, indices in enumerate(test)])
    summary['Total'] = summary.sum(axis=1)
    # summary.columns = summary.columns.rename('')
    ICD.display(summary)