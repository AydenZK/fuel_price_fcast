import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf
from pmdarima.arima.utils import ndiffs

FIGSIZE = (16, 8)

def plot_seasonality(df, col, date_col='date'):
    df = df.copy()
    df['year'] = df[date_col].apply(lambda x: x.year)
    df['month'] = df[date_col].apply(lambda x: x.month)
    df['dayofweek'] = df[date_col].apply(lambda x: x.day_name())

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=FIGSIZE)

    palette = sns.color_palette("ch:2.5,-.2,dark=.3", df['year'].nunique())
    sns.lineplot(df, x = 'month', y = col, hue='year', palette=palette, ax=ax[0])
    ax[0].set_title(f'Seasonal plot of {col}', fontsize = 20, loc='center', fontdict=dict(weight='bold'))
    ax[0].set_xlabel('Month', fontsize = 16, fontdict=dict(weight='bold'))
    ax[0].set_ylabel(col, fontsize = 16, fontdict=dict(weight='bold'))

    sns.boxplot(df, x = 'year', y = col, ax=ax[1])
    ax[1].set_title('Year-wise Box Plot\n(The Trend)', fontsize = 20, loc='center', fontdict=dict(weight='bold'))
    ax[1].set_xlabel('Year', fontsize = 16, fontdict=dict(weight='bold'))
    ax[1].set_ylabel(col, fontsize = 16, fontdict=dict(weight='bold'))

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=FIGSIZE)
    sns.boxplot(df, x = 'month', y = col, ax=ax[0])
    ax[0].set_title('Month-wise Box Plot\n(The Seasonality)', fontsize = 20, loc='center', fontdict=dict(weight='bold'))
    ax[0].set_xlabel('Month', fontsize = 16, fontdict=dict(weight='bold'))
    ax[0].set_ylabel(col, fontsize = 16, fontdict=dict(weight='bold'))

    sns.boxplot(df, x = 'dayofweek', y = col, ax=ax[1])
    ax[1].set_title('Day-wise Box Plot\n(The Seasonality)', fontsize = 20, loc='center', fontdict=dict(weight='bold'))
    ax[1].set_xlabel('Day', fontsize = 16, fontdict=dict(weight='bold'))
    ax[1].set_ylabel(col, fontsize = 16, fontdict=dict(weight='bold'))

def plot_decomposition(df, col, period=365):
    # Multiplicative Decomposition 
    multiplicative_decomposition = seasonal_decompose(df[col], model='multiplicative', period=365)

    # Additive Decomposition
    additive_decomposition = seasonal_decompose(df[col], model='additive', period=365)

    # Plot
    plt.rcParams.update({'figure.figsize': FIGSIZE})
    multiplicative_decomposition.plot().suptitle('Multiplicative Decomposition', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    additive_decomposition.plot().suptitle('Additive Decomposition', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    plt.show()

def plot_differencing(df, col):
    plt.rcParams.update({'figure.figsize':FIGSIZE, 'figure.dpi':120})

    # Original Series
    fig, axes = plt.subplots(3, 2, sharex=True)
    axes[0, 0].plot(df[col]); axes[0, 0].set_title('Original Series')
    plot_acf(df[col], ax=axes[0, 1])

    # 1st Differencing
    axes[1, 0].plot(df[col].diff()); axes[1, 0].set_title('1st Order Differencing')
    plot_acf(df[col].diff().dropna(), ax=axes[1, 1])

    # 2nd Differencing
    axes[2, 0].plot(df[col].diff().diff()); axes[2, 0].set_title('2nd Order Differencing')
    plot_acf(df[col].diff().diff().dropna(), ax=axes[2, 1])

    plt.show()

    print('Recommended Differencing:')
    print(f'ADF: {ndiffs(df[col], test="adf")}')
    print(f'KPSS: {ndiffs(df[col], test="kpss")}')
    print(f'PP: {ndiffs(df[col], test="pp")}')