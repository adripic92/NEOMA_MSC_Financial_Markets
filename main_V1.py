#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Created on Mon Nov 25 19:04:22 2024

@author: adrienpicard

Comprehensive Portfolio Optimization and Visualization

Plots stocks, risk-free asset, GMVP, MSR, Efficient Frontier, and CML with proper labeling.
Prints detailed portfolio metrics, including returns, volatilities, correlations, and weights.
"""



import yfinance as yf 
import numpy as np
import pandas as pd
from pandas_datareader import data as pdr

import scipy.optimize as sco
import statsmodels.api as sm

import matplotlib.pyplot as plt
import seaborn as sns


from datetime import datetime
import time
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Constants
TICKERS = ['ML.PA', 'HDB', 'CEG', 'RTX', '8031.T', 'INVALID'] #INVALID is to prove our error handling (parsing of tickers)
START_DATE = "2019-01-01" # 5 years of data
END_DATE = datetime.today().strftime('%Y-%m-%d') #Most recent data fetching
RISK_FREE_RATE = 0.05  # Risk-free rate as a decimal (5%)


def timed_input(prompt, timeout=8, default="yes"):
    """
    Prompt user for input with a timeout. Returns default value if no input is given within the timeout.
    """
    print(f"{prompt} (You have {timeout} seconds to respond)")
    start_time = time.time()
    response = ''
    while time.time() - start_time < timeout and not response:
        if response := input().strip().lower():
            break
    return response if response else default

def compute_stock_metrics(tickers, start_date, end_date):
    valid_tickers = []
    data_dict = {}
    company_names = []
    
    # Validate tickers and download data
    for ticker in tickers:
        try:
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)['Adj Close']
            if not data.empty:
                valid_tickers.append(ticker)
                data_dict[ticker] = data
                # Fetch company name
                try:
                    stock_info = yf.Ticker(ticker).info
                    company_name = stock_info.get('shortName', ticker)
                except Exception:
                    company_name = ticker  # Fallback to ticker symbol
                company_names.append(company_name)
            else:
                print(f"Skipping {ticker}: No data available.")
        except Exception as e:
            print(f"Skipping {ticker}: {e}")

    if not valid_tickers:
        raise ValueError("No valid data downloaded for the provided tickers.")
    
    # Combine data into a single DataFrame
    data = pd.DataFrame(data_dict)
    data.index = pd.to_datetime(data.index)
    daily_returns = data.pct_change()

    # Initialize lists to store metrics
    returns = []
    volatilities = []

    for ticker in valid_tickers:
        if data[ticker].isnull().all():
            returns.append(None)
            volatilities.append(None)
            continue

        # Calculate annualized return and volatility
        start_value = data[ticker].dropna().iloc[0]
        end_value = data[ticker].dropna().iloc[-1]
        years = (data[ticker].dropna().index[-1] - data[ticker].dropna().index[0]).days / 365.25

        annualized_return = ((end_value / start_value) ** (1 / years) - 1) * 100
        annualized_volatility = daily_returns[ticker].std() * np.sqrt(252) * 100

        returns.append(annualized_return)
        volatilities.append(annualized_volatility)

    # Compute the mean and thresholds for extreme values
    mean_return = np.nanmean(returns)
    mean_volatility = np.nanmean(volatilities)
    return_threshold = mean_return * 1.8
    volatility_threshold = mean_volatility * 1.8

    # Filter out extreme values and prompt user decision
    final_tickers = []
    final_returns = []
    final_volatilities = []
    final_company_names = []

    for i, ticker in enumerate(valid_tickers):
        if returns[i] is None or volatilities[i] is None:
            continue

        # Check if return or volatility exceeds the threshold
        is_extreme = returns[i] > return_threshold or volatilities[i] > volatility_threshold
        if is_extreme:
            print(f"Ticker {ticker} has extreme values: Return={returns[i]:.2f}%, Volatility={volatilities[i]:.2f}%")
            response = timed_input("Do you want to include this stock in the analysis? (yes/no): ", timeout=5)
            if response != "yes":
                print(f"Excluding {ticker} from the analysis.")
                continue

        # Add ticker and corresponding name only if user accepts or it's not extreme
        final_tickers.append(ticker)
        final_returns.append(returns[i])
        final_volatilities.append(volatilities[i])
        final_company_names.append(company_names[i])

    # Create correlation matrix and validate covariance matrix
    if final_tickers:
        daily_returns = daily_returns[final_tickers].dropna(how="all", axis=0)  # Drop rows with all NaN
        correlation_matrix = daily_returns.corr()
        cov_matrix = daily_returns.cov() * 252  # Annualized covariance matrix
        
        # Validate covariance matrix
        valid_assets = np.all(np.isfinite(cov_matrix), axis=0) & (np.array(final_volatilities) > 0)
        cov_matrix = cov_matrix.loc[valid_assets, valid_assets]
        final_tickers = [final_tickers[i] for i, valid in enumerate(valid_assets) if valid]
        final_returns = [final_returns[i] for i, valid in enumerate(valid_assets) if valid]
        final_volatilities = [final_volatilities[i] for i, valid in enumerate(valid_assets) if valid]
        final_company_names = [final_company_names[i] for i, valid in enumerate(valid_assets) if valid]
    else:
        correlation_matrix = pd.DataFrame()
        cov_matrix = pd.DataFrame()

    # Create DataFrame for output
    metrics_df = pd.DataFrame({
        "Stock_Ticker": final_tickers,
        "Company_Name": final_company_names,
        "Annual_Return%": final_returns,
        "Annual_Volatility%": final_volatilities,
    })

    return {
        "Metrics_DataFrame": metrics_df,
        "Correlation_Matrix": correlation_matrix,
        "Covariance_Matrix": cov_matrix,
    }

# Function to compute portfolio metrics
def portfolio_metrics(weights, returns, cov_matrix):
    portfolio_return = np.dot(weights, returns)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe_ratio = (portfolio_return - RISK_FREE_RATE) / portfolio_volatility
    return portfolio_return, portfolio_volatility, sharpe_ratio


# Function to calculate the global minimum variance portfolio (GMVP)
def optimize_min_risk(mean_returns, cov_matrix):
    num_assets = len(mean_returns)

    def portfolio_volatility(weights):
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

    constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}
    bounds = tuple((0.0, 1.0) for _ in range(num_assets))

    result = sco.minimize(
        portfolio_volatility,
        num_assets * [1.0 / num_assets],
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
    )

    if result.success:
        return result.x, result.fun
    else:
        raise ValueError("Optimization failed")

# Function to optimize for maximum Sharpe Ratio (MSR)
def optimize_sharpe_ratio(mean_returns, cov_matrix):
    num_assets = len(mean_returns)

    def negative_sharpe_ratio(weights):
        portfolio_return, portfolio_volatility, _ = portfolio_metrics(weights, mean_returns, cov_matrix)
        return -(portfolio_return - RISK_FREE_RATE) / portfolio_volatility

    constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}
    bounds = tuple((0.0, 1.0) for _ in range(num_assets))

    result = sco.minimize(
        negative_sharpe_ratio,
        num_assets * [1.0 / num_assets],
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
    )

    if result.success:
        return result.x, result.fun
    else:
        raise ValueError("Optimization failed for Sharpe Ratio")

# Function to print portfolio weights in a readable format
def print_portfolio_weights(weights, tickers):
    print("\nPortfolio Weights:")
    print(f"{'Ticker':<10}{'Weight (%)':<15}")
    print("-" * 25)
    for ticker, weight in zip(tickers, weights):
        print(f"{ticker:<10}{weight * 100:<15.2f}")

# Function to calculate the efficient frontier
def calculate_efficient_frontier(mean_returns, cov_matrix, num_points=100):
    target_returns = np.linspace(mean_returns.min(), mean_returns.max(), num_points)
    efficient_volatilities = []

    for target_return in target_returns:
        constraints = (
            {"type": "eq", "fun": lambda weights: np.sum(weights) - 1},
            {"type": "eq", "fun": lambda weights: np.dot(weights, mean_returns) - target_return},
        )
        bounds = [(0.0, 1.0) for _ in range(len(mean_returns))]
        result = sco.minimize(
            lambda weights: np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))),
            np.ones(len(mean_returns)) / len(mean_returns),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
        )
        if result.success:
            efficient_volatilities.append(result.fun * 100)  # Convert to %
        else:
            efficient_volatilities.append(np.nan)

    target_returns = target_returns[~np.isnan(efficient_volatilities)]
    efficient_volatilities = np.array(efficient_volatilities)[~np.isnan(efficient_volatilities)]

    return target_returns * 100, efficient_volatilities


# Function to simulate random portfolios 2000 per default (WARNING could take time if you set higher than 100 000)
def simulate_portfolios(returns, cov_matrix, num_portfolios=2000):
    portfolio_returns = []
    portfolio_volatilities = []
    sharpe_ratios = []

    for _ in range(num_portfolios):
        weights = np.random.random(len(returns))
        weights /= np.sum(weights)

        annual_return = np.dot(weights, returns)
        volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe_ratio = (annual_return - RISK_FREE_RATE) / volatility

        portfolio_returns.append(annual_return * 100)
        portfolio_volatilities.append(volatility * 100)
        sharpe_ratios.append(sharpe_ratio)

    return np.array(portfolio_returns), np.array(portfolio_volatilities), np.array(sharpe_ratios)

def plot_historical_prices(metrics_df, start_date, end_date, base_currency="USD", title="Historical Prices (Converted)"):
    tickers = metrics_df['Stock_Ticker'].tolist()
    company_names = metrics_df['Company_Name'].tolist()
    exchange_rates = {}  # Dictionary to store currency conversion rates
    stock_currencies = {}  # Dictionary to store stock currencies

    # Retrieve currency for each stock
    for ticker in tickers:
        try:
            stock_info = yf.Ticker(ticker).info
            stock_currencies[ticker] = stock_info.get('currency', 'Unknown')
        except Exception as e:
            print(f"Error retrieving currency for {ticker}: {e}")
            stock_currencies[ticker] = 'Unknown'

    # Example hardcoded exchange rates (you can replace these with API calls)
    conversion_rates = {
        "USD": 1.0,    # Base currency
        "EUR": 1.1,    # 1 EUR = 1.1 USD
        "JPY": 0.008,  # 1 JPY = 0.008 USD
        "GBP": 1.25,   # 1 GBP = 1.25 USD
        # Add more currencies as needed
    }

    # Validate and populate exchange rates for all stock currencies
    for ticker, currency in stock_currencies.items():
        if currency in conversion_rates:
            exchange_rates[ticker] = conversion_rates[currency]
        else:
            print(f"Missing conversion rate for {currency}. Skipping {ticker}.")
            exchange_rates[ticker] = None

    # Download historical data
    data_dict = {}
    for ticker in tickers:
        try:
            #print(f"Downloading data for {ticker}...")
            data = yf.download(ticker, start=start_date, end=end_date,progress=False)['Adj Close']
            if not data.empty:
                # Apply currency conversion
                if exchange_rates[ticker] is not None:
                    data = data * exchange_rates[ticker]
                data_dict[ticker] = data
            else:
                print(f"No data available for {ticker}. Skipping...")
        except Exception as e:
            print(f"Error downloading data for {ticker}: {e}")

    # Combine data into a single DataFrame
    if not data_dict:
        print("No valid data available for any tickers.")
        return

    data = pd.DataFrame(data_dict)

    # Align data columns with tickers
    data = data.rename(columns=lambda x: x.strip())
    missing_tickers = [ticker for ticker in tickers if ticker not in data.columns]
    if missing_tickers:
        print(f"Warning: Missing tickers in data: {missing_tickers}")
    # Set up the plot
    plt.figure(figsize=(14, 8), dpi=300)
    sns.set_style("whitegrid")

    # Plot each stock in the converted currency
    for ticker, name in zip(tickers, company_names):
        if ticker in data.columns:
            plt.plot(
                data.index,
                data[ticker],
                label=f"{name} ({ticker})",
                linewidth=2,
            )
        else:
            print(f"No data to plot for {name} ({ticker}). Skipping...")

    # Add title, labels, and grid
    plt.title(title, fontsize=16, pad=20)
    plt.xlabel("Date", fontsize=14)
    plt.ylabel(f"Adjusted Price ({base_currency})", fontsize=14)
    plt.grid(alpha=0.5)

    # Add legend in the top-left corner inside the graph
    plt.legend(
        title="Stocks",
        fontsize=10,
        loc="upper left",  # Place legend in the top-left corner
        bbox_to_anchor=(0.01, 0.99),  # Slightly inset from the top-left corner
        borderaxespad=0,
        frameon=True,  # Display a frame around the legend
        fancybox=True,  # Rounded edges for the frame
        framealpha=0.6,  # Semi-transparent frame
    )

    # Add a note about currencies
    plt.figtext(
        0.5, -0.05,
        f"Note: Prices are converted to {base_currency} using static exchange rates; stocks with no data are skipped.",
        wrap=True, horizontalalignment='center', fontsize=10, color="gray"
    )

    # Improve layout
    plt.tight_layout()
    plt.show()


def plot_correlation_heatmap(correlation_matrix, tickers, title="Correlation Heatmap"):
 
    plt.figure(figsize=(10, 8), dpi=300)
    sns.heatmap(
        correlation_matrix, 
        annot=True, 
        cmap="coolwarm",
        fmt=".2f", 
        xticklabels=[f"{ticker}" for ticker in tickers],
        yticklabels=[f"{ticker}" for ticker in tickers],
        linewidths=0.5, 
        cbar_kws={'label': 'Correlation Coefficient'}
    )
    plt.title(title, fontsize=16, pad=20)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    plt.show()

def plot_gmvp_msr_weights(tickers, company_names, gmvp_weights, msr_weights, gmvp_performance, msr_performance):
    gmvp_return, gmvp_volatility = gmvp_performance
    msr_return, msr_volatility = msr_performance

    # Retrieve company names
    labels = [f"{name} ({ticker})" for name, ticker in zip(company_names, tickers)]
    x = np.arange(len(labels))  # Label positions
    width = 0.1  # Reduced bar width for closer bars
    offset = 0.15

    fig, ax = plt.subplots(figsize=(12, 8), dpi=300)

    # GMVP Weights
    bars_gmvp = ax.bar(x - offset, gmvp_weights * 100, width, label='GMVP Weights', color='red', edgecolor='black')
    # MSR Weights
    bars_msr = ax.bar(x + offset, msr_weights * 100, width, label='MSR Weights', color='blue', edgecolor='black')

    # Add percentages on GMVP bars
    for bar, weight in zip(bars_gmvp, gmvp_weights * 100):
        ax.text(
            bar.get_x() + bar.get_width() / 2,  # Center of the bar
            bar.get_height() + 1,  # Slightly above the bar
            f"{weight:.1f}%",  # Format the percentage
            ha='center', va='bottom', fontsize=10, color='black'
        )

    # Add percentages on MSR bars
    for bar, weight in zip(bars_msr, msr_weights * 100):
        ax.text(
            bar.get_x() + bar.get_width() / 2,  # Center of the bar
            bar.get_height() + 1,  # Slightly above the bar
            f"{weight:.1f}%",  # Format the percentage
            ha='center', va='bottom', fontsize=10, color='black'
        )

    # Adding text annotations for performance
    # GMVP Annotation
    ax.text(
        len(labels) - 1,  # Position near the last bar
        max(max(gmvp_weights), max(msr_weights)) * 100 - 15,  # Lower placement
        f"GMVP:\nReturn={gmvp_return * 100:.2f}%,\nVolatility={gmvp_volatility * 100:.2f}%",
        color="red",
        fontsize=8,  # Smaller font size
        ha='center',  # Horizontal alignment
        bbox=dict(facecolor="white", alpha=0.4, edgecolor="gray")  # Less intrusive box
    )

    # MSR Annotation
    ax.text(
        len(labels) - 1,  # Position near the last bar
        max(max(gmvp_weights), max(msr_weights)) * 100 - 25,  # Lower placement
        f"MSR:\nReturn={msr_return * 100:.2f}%,\nVolatility={msr_volatility * 100:.2f}%",
        color="blue",
        fontsize=8,  # Smaller font size
        ha='center',  # Horizontal alignment
        bbox=dict(facecolor="white", alpha=0.4, edgecolor="gray")  # Less intrusive box
    )

    # Titles and labels
    ax.set_title("Portfolio Weights for GMVP and MSR", fontsize=16, pad=20)
    ax.set_xlabel("Stocks", fontsize=14)
    ax.set_ylabel("Weight (%)", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10, rotation=45, ha='right')  # Set smaller font size here
    ax.legend(fontsize=12)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()
 
def plot_return_vs_risk_portfolio_GMVP_MSR(
        stock_df,
        simulated_returns,
        simulated_volatilities,
        simulated_sharpe_ratios,
        efficient_returns,
        efficient_volatilities,
        msr_return,
        msr_volatility,
        min_risk_return,
        min_risk_volatility):
    
    plt.figure(figsize=(14, 10), dpi=300)
    sns.set_style("whitegrid")
    #plt.rcParams['font.family'] = 'DejaVu Sans'  # High-quality font     
    
    # Simulated Portfolios
    scatter = plt.scatter(simulated_volatilities, simulated_returns, c=simulated_sharpe_ratios, cmap="viridis", alpha=0.6, s=10)
    plt.colorbar(scatter, label="Sharpe Ratio",shrink=1, aspect=30, pad=0.02)

    # Risk-Free Rate
    plt.scatter(0, RISK_FREE_RATE * 100, color="lime", label="Risk-Free Rate", edgecolor="Teal", s=150, zorder=3)
    plt.text(-0.2, RISK_FREE_RATE * 100 + 8, "Risk-Free\n(5%)", fontsize=10, ha="center",
         bbox=dict(facecolor="white", edgecolor="gray", boxstyle="round,pad=0.3"))

    # Stock Points
    for _, row in stock_df.iterrows():
        plt.scatter(row["Stock_volatility%"], row["Stock_return%"], color="orange", edgecolor="black", s=100, zorder=3)
        plt.text(row["Stock_volatility%"] + 0.5, row["Stock_return%"] + 2,
             f"{row['Stock_name']}\n({row['Stock_return%']:.1f}%)", fontsize=9,
             bbox=dict(facecolor="white", edgecolor="gray", boxstyle="round,pad=0.3"))
   
    # Plot Dotted Lines from Stocks to Portfolio
    for _, row in stock_df.iterrows():
        plt.plot(
            [row["Stock_volatility%"], min_risk_volatility * 100],
            [row["Stock_return%"], min_risk_return * 100],
            linestyle="dotted", color="gray"
            )

    # GMVP
    plt.scatter(min_risk_volatility * 100, min_risk_return * 100, color="red",edgecolor="gray", label="GMVP (Minimum Risk)", marker="H", s=180, zorder=3)
    plt.text(min_risk_volatility * 100 - 3.2, min_risk_return * 100 + 2, f"GMVP\n({min_risk_return * 100:.2f}%)",
         fontsize=10, bbox=dict(facecolor="white", edgecolor="gray", boxstyle="round,pad=0.3"))

    # MSR
    plt.scatter(msr_volatility * 100, msr_return * 100, color="blue", label="MSR (Maximum Sharpe Ratio) Portfolio", marker="o", s=180,zorder=3)
    plt.text(msr_volatility * 100 - 3, msr_return * 100 + 5, f"MSR\n({msr_return * 100:.2f}%)",
         fontsize=10, bbox=dict(facecolor="white", edgecolor="gray", boxstyle="round,pad=0.3"))

    # Efficient Frontier
    plt.plot(efficient_volatilities, efficient_returns, color="blue", linestyle="--", linewidth=1.5, label="Efficient Frontier")
    
    # Add Axis Ticks
    plt.xticks(np.arange(0, max(efficient_volatilities) + 1, 2))  # Set ticks up to 26 for risk
    plt.yticks(np.arange(0, max(efficient_returns) + 1, 5))  # Set ticks up to 52 for return
    
    # Capital Market Line (CML)
    cml_x = np.linspace(0, msr_volatility * 100, 100)
    cml_y = RISK_FREE_RATE * 100 + (msr_return * 100 - RISK_FREE_RATE * 100) * (cml_x / (msr_volatility * 100))
    plt.plot(cml_x, cml_y, label="Capital Market Line (CML)", color="green", linestyle="-", linewidth=1.5)
   
    # Titles, labels, and legend
    plt.title("Expected Return vs Risk for Portfolio and Individual Assets", fontsize=18, pad=20)
    plt.xlabel("Risk (Volatility %)", fontsize=14)
    plt.ylabel("Expected Return (%)", fontsize=14)
    plt.grid(alpha=0.4)
    plt.legend(fontsize=12)
    plt.tight_layout()
    #plt.savefig("high_quality_plot.png", dpi=300, bbox_inches='tight', transparent=False)  # Save as high-res PNG
    #plt.savefig("high_quality_plot.svg", format="svg")
    plt.show()

def fetch_fama_french_factors(start_date, end_date):
    """
    Fetches the Fama-French 3-factor daily data and processes it for the given date range.
    """
    #print("Downloading Fama-French 3-factor data...")
    ff3_data = pdr.DataReader('F-F_Research_Data_Factors_daily', 'famafrench', start_date, end_date)[0]
    ff3_data.index = pd.to_datetime(ff3_data.index)  # Convert to datetime index
    
    # Debug: Print available columns
    print("Fama-French Data Columns:", ff3_data.columns)
    
    # Rename columns for consistency
    ff3_data = ff3_data.rename(columns={
        'Mkt-RF': 'MKT-RF',  # Adjust column names as needed
        'HML': 'HML',
        'SMB': 'SMB',
        'RF': 'RF'
    })
    ff3_data = ff3_data / 100  # Convert from percentages to decimals
    return ff3_data


def compute_fama_french_3_factors(portfolio_returns, ff3_factors, portfolio_name="Portfolio"):
    """
    Computes and visualizes the Fama-French 3-Factor Model for a portfolio or stock.
    Prints summary statistics for 'MKT-RF', 'SMB', 'HML', and 'RF'.

    Args:
        portfolio_returns (pd.Series): The portfolio's excess returns (aligned with Fama-French data).
        ff3_factors (pd.DataFrame): Fama-French factors ('MKT-RF', 'SMB', 'HML', and 'RF').
        portfolio_name (str): Name of the portfolio/stock for labeling.
    """
    # Align data
    common_dates = portfolio_returns.index.intersection(ff3_factors.index)
    portfolio_returns = portfolio_returns.loc[common_dates]
    ff3_factors = ff3_factors.loc[common_dates]

    # Calculate portfolio excess returns
    portfolio_excess_returns = portfolio_returns - ff3_factors['RF']

    # Prepare regression data
    X = ff3_factors[['MKT-RF', 'SMB', 'HML']]
    X = sm.add_constant(X)  # Add intercept term
    y = portfolio_excess_returns

    # Perform regression
    model = sm.OLS(y, X).fit()

    # Print regression summary
    print(model.summary())

    # Visualize coefficients
    factors = ['Alpha (Intercept)', 'MKT-RF', 'SMB', 'HML']
    coefficients = model.params

    plt.figure(figsize=(10, 6))
    plt.bar(factors, coefficients, color=['blue', 'orange', 'green', 'red'], edgecolor='black')
    plt.title(f"Fama-French 3-Factor Coefficients for {portfolio_name}", fontsize=14)
    plt.ylabel("Coefficient Value", fontsize=12)
    plt.xlabel("Factors", fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Scatterplot of fitted vs actual excess returns
    plt.figure(figsize=(10, 6))
    plt.scatter(y, model.fittedvalues, alpha=0.7, label="Fitted vs Actual")
    plt.plot([y.min(), y.max()], [y.min(), y.max()], color="red", linestyle="--", label="45° Line")
    plt.title(f"Fitted vs Actual Excess Returns for {portfolio_name}", fontsize=14)
    plt.xlabel("Actual Excess Returns", fontsize=12)
    plt.ylabel("Fitted Excess Returns", fontsize=12)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Print Fama-French Factor Ratios
    print("\n=== Fama-French Factor Ratios ===")
    factor_stats = pd.DataFrame({
        'Mean': ff3_factors.mean(),
        'Std Dev': ff3_factors.std(),
        'Correlation with Portfolio Excess Returns': ff3_factors.corrwith(portfolio_excess_returns)
    }).loc[['MKT-RF', 'SMB', 'HML', 'RF']]  # Select only relevant factors

    # Display the calculated statistics
    print(factor_stats)

    # Optional: Plot factor statistics (mean and standard deviation)
    factor_stats[['Mean', 'Std Dev']].plot(kind='bar', figsize=(10, 6), edgecolor='black')
    plt.title("Fama-French Factor Mean and Standard Deviation", fontsize=14)
    plt.ylabel("Value", fontsize=12)
    plt.xlabel("Factors", fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_sml_with_dynamic_gmvp(metrics_df, gmvp_weights, market_ticker='URTH', start_date=START_DATE, end_date=END_DATE):
    """
    Plots the Security Market Line (SML) and includes the dynamically computed GMVP.

    Args:
        metrics_df (pd.DataFrame): DataFrame with stock metrics (returns, tickers, etc.).
        gmvp_weights (list): Weights of the GMVP portfolio.
        market_ticker (str): Ticker for the market proxy (e.g., MSCI World: 'URTH').
        start_date (str): Start date for analysis.
        end_date (str): End date for analysis.
    """
    #print(f"Downloading market data for {market_ticker}...")
    try:
        market_data = yf.download(market_ticker, start=start_date, end=end_date, progress=False)['Adj Close']
        if market_data.empty:
            raise ValueError(f"No data found for {market_ticker}. Check the ticker.")
    except Exception as e:
        print(f"Error fetching market data: {e}")
        return

    # Step 1: Calculate market returns
    market_daily_returns = market_data.pct_change().dropna()
    market_annual_return = ((1 + market_daily_returns.mean()) ** 252) - 1

    # Step 2: Calculate stock betas via regression
    stock_betas = []
    for _, row in metrics_df.iterrows():
        stock_ticker = row['Stock_Ticker']
        #print(f"Calculating beta for {stock_ticker}...")
        stock_data = yf.download(stock_ticker, start=start_date, end=end_date, progress=False)['Adj Close'].pct_change().dropna()

        # Align dates
        common_dates = stock_data.index.intersection(market_daily_returns.index)
        stock_data = stock_data.loc[common_dates]
        market_returns = market_daily_returns.loc[common_dates]

        # Regression to calculate beta
        X = sm.add_constant(market_returns)
        model = sm.OLS(stock_data, X).fit()
        stock_betas.append(model.params[1])  # Beta is the slope coefficient

    metrics_df['Beta'] = stock_betas

    # Step 3: Compute GMVP beta
    gmvp_beta = np.dot(gmvp_weights, metrics_df['Beta'])
    gmvp_return = np.dot(gmvp_weights, metrics_df['Annual_Return%']) / 100  # Convert to decimal for calculations

    # Step 4: Generate SML
    betas = np.linspace(-0.5, 2.5, 100)
    sml_returns = RISK_FREE_RATE + betas * (market_annual_return - RISK_FREE_RATE)

    # Step 5: Plot the SML
    plt.figure(figsize=(12, 8))
    plt.plot(betas, sml_returns * 100, label="Security Market Line (SML)", linestyle="--", color="blue", linewidth=2)

    # Plot GMVP point
    plt.scatter(gmvp_beta, gmvp_return * 100, color="red", label=f"GMVP ({gmvp_return:.2%})", s=100, zorder=5)
    plt.text(gmvp_beta + 0.05, gmvp_return * 100 + 0.5, f"GMVP ({gmvp_return:.2%})", fontsize=10)

    # Plot individual stocks
    for _, row in metrics_df.iterrows():
        stock_return = row['Annual_Return%'] / 100
        stock_beta = row['Beta']
        plt.scatter(stock_beta, stock_return * 100, color='orange', zorder=5)
        plt.text(stock_beta + 0.05, stock_return * 100 + 0.5, row['Stock_Ticker'], fontsize=8)

    # Add Risk-Free Rate Line
    plt.axhline(y=RISK_FREE_RATE * 100, color="green", linestyle="-", label="Risk-Free Rate")

    # Enhance chart aesthetics
    plt.title("Security Market Line (SML) with GMVP", fontsize=16)
    plt.xlabel("Beta (Systematic Risk)", fontsize=14)
    plt.ylabel("Expected Return (%)", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    # Display the plot
    plt.show()
    
def evaluate_gmvp_performance(gmvp_weights, gmvp_return, gmvp_volatility, cov_matrix, risk_free_rate, market_ticker, start_date, end_date):
    """
    Evaluates the performance of the Global Minimum Variance Portfolio (GMVP).
    """
    print(f"Fetching market data for benchmark: {market_ticker}")
    market_data = yf.download(market_ticker, start=start_date, end=end_date, progress=False)["Adj Close"]
    market_log_returns = np.log(market_data / market_data.shift(1)).dropna()

    prices = pd.DataFrame({ticker: yf.download(ticker, start=start_date, end=end_date,progress=False)["Adj Close"] for ticker in TICKERS})
    prices = prices.replace(0, np.nan)  # Handle zero prices

    log_returns = np.log(prices / prices.shift(1)).dropna()
    aligned_data = log_returns.join(market_log_returns.rename("Market"), how="inner")
    aligned_data.replace([np.inf, -np.inf], np.nan, inplace=True)  # Handle invalid values
    aligned_data.dropna(inplace=True)

    # Debugging
    print(f"Cleaned Aligned Data Shape: {aligned_data.shape}")
    print(f"Cleaned Aligned Data Sample:\n{aligned_data.head()}")

    # Check for market variance
    market_variance = aligned_data["Market"].var()
    if market_variance == 0 or np.isnan(market_variance):
        print("Error: Market returns have zero variance or invalid data.")
        return {"Error": "Market returns variance is zero or invalid."}

    # Calculate portfolio beta
    betas = [
        np.cov(aligned_data[stock], aligned_data["Market"])[0, 1] / market_variance
        if market_variance > 0 else np.nan
        for stock in TICKERS
    ]
    portfolio_beta = np.dot(gmvp_weights, betas)

    # Debugging: Print calculated betas
    print(f"Stock Betas: {betas}")
    print(f"Portfolio Beta: {portfolio_beta}")

    # Market annualized return and volatility
    annualized_market_return = ((1 + market_log_returns.mean()) ** 252) - 1
    annualized_market_volatility = market_log_returns.std() * np.sqrt(252)

    # Jensen's Alpha
    jensens_alpha = gmvp_return - (risk_free_rate + portfolio_beta * (annualized_market_return - risk_free_rate))

    performance_metrics = {
        "GMVP Annualized Return (%)": gmvp_return * 100,
        "GMVP Annualized Volatility (%)": gmvp_volatility * 100,
        "GMVP Sharpe Ratio": (gmvp_return - risk_free_rate) / gmvp_volatility,
        "GMVP Beta": portfolio_beta,
        "GMVP Jensen's Alpha": jensens_alpha,
        "Market Annualized Return (%)": annualized_market_return * 100,
        "Market Annualized Volatility (%)": annualized_market_volatility * 100
    }

    # Print results
    print("\n=== GMVP Performance Metrics ===")
    for key, value in performance_metrics.items():
        print(f"{key}: {value:.4f}" if isinstance(value, (float, int)) else f"{key}: {value}")

    return performance_metrics



# Main function
def main():
    stock_metrics = compute_stock_metrics(TICKERS, START_DATE, END_DATE)
    
    # Extract results
    metrics_df = stock_metrics["Metrics_DataFrame"]
    correlation_matrix = stock_metrics["Correlation_Matrix"]
    cov_matrix = stock_metrics["Covariance_Matrix"]

    # Check if there are valid assets
    if metrics_df.empty:
        print("No valid stocks remaining after filtering.")
        return

    # Portfolio optimization
    returns = metrics_df["Annual_Return%"].values / 100
    tickers = metrics_df["Stock_Ticker"].values

    num_assets = len(returns)
    
    # Check if there are valid assets
    if num_assets == 0:
        print("No valid assets remaining after filtering. Ensure input data is correct.")
        return

    try:
        # GMVP Calculation
        gmvp_weights, gmvp_volatility = optimize_min_risk(returns, cov_matrix)
        gmvp_return = np.dot(gmvp_weights, returns)

        # MSR Calculation
        msr_weights, _ = optimize_sharpe_ratio(returns, cov_matrix)
        msr_return, msr_volatility, _ = portfolio_metrics(msr_weights, returns, cov_matrix)

        # Print Stock Performance
        print("\nTicker    Return (%)     Volatility (%)")
        print("-" * 45)
        for ticker, ret, vol in zip(tickers, returns * 100, metrics_df['Annual_Volatility%']):
            print(f"{ticker:<10}{ret:<15.2f}{vol:<20.2f}")

        # Print Correlation Matrix
        correlation_df = pd.DataFrame(correlation_matrix, index=tickers, columns=tickers)
        print("\n=== Correlation Matrix ===")
        print(correlation_df)

        # Print Portfolio Metrics
        print("\n=== Portfolio Metrics ===")
        print(f"\nGMVP Return: {gmvp_return * 100:.2f}%, GMVP Volatility: {gmvp_volatility * 100:.2f}%")
        print_portfolio_weights(gmvp_weights, tickers)
        print(f"\n\nMSR Return: {msr_return * 100:.2f}%, MSR Volatility: {msr_volatility * 100:.2f}%")
        print_portfolio_weights(msr_weights, tickers)

        # Efficient Frontier
        efficient_returns, efficient_volatilities = calculate_efficient_frontier(returns, cov_matrix)

        # Simulate Portfolios
        simulated_returns, simulated_volatilities, simulated_sharpe_ratios = simulate_portfolios(returns, cov_matrix)
        
        # Create DataFrame for stock results
        stock_df = pd.DataFrame({
            "Stock_name": metrics_df['Stock_Ticker'],
            "Stock_return%": metrics_df['Annual_Return%'],
            "Stock_volatility%": metrics_df["Annual_Volatility%"]
            })
        # Evaluate GMVP performance
        evaluate_gmvp_performance(
            gmvp_weights=gmvp_weights,
            gmvp_return=gmvp_return,
            gmvp_volatility=gmvp_volatility,
            cov_matrix=cov_matrix,
            risk_free_rate=RISK_FREE_RATE,
            market_ticker="URTH",  # MSCI World as the benchmark
            start_date=START_DATE,
            end_date=END_DATE
        )
        
        # Plot Correlation Heatmap
        plot_correlation_heatmap(correlation_df, tickers, title="Correlation Heatmap for Selected Stocks")
    
        #plot main graph (CML / Optimal Portfolio)
        plot_return_vs_risk_portfolio_GMVP_MSR(
        stock_df,
        simulated_returns,
        simulated_volatilities,
        simulated_sharpe_ratios,
        efficient_returns,
        efficient_volatilities,
        msr_return,
        msr_volatility,
        gmvp_return,
        gmvp_volatility
        )
        
        # Plot GMVP and MSR Portfolio Weights
        plot_gmvp_msr_weights(
            metrics_df['Stock_Ticker'],
            metrics_df['Company_Name'],
            gmvp_weights,
            msr_weights,
            (gmvp_return, gmvp_volatility),
            (msr_return, msr_volatility)
        )
        
        plot_historical_prices(metrics_df, START_DATE, END_DATE, base_currency="USD", title="Historical Prices in USD")
        
        # Compute and visualize Fama-French model
        portfolio_daily_returns = metrics_df['Annual_Return%'] / 252  # Approximate daily returns from annual returns
        portfolio_returns = pd.Series(portfolio_daily_returns.values, index=pd.date_range(start=START_DATE, periods=len(portfolio_daily_returns), freq='B')) # Align the portfolio returns with date index# 
        compute_fama_french_3_factors(portfolio_returns, fetch_fama_french_factors(START_DATE, END_DATE), portfolio_name="Sample Portfolio")
       
        # Compute the SML grap
        plot_sml_with_dynamic_gmvp(metrics_df, gmvp_weights, start_date=START_DATE, end_date=END_DATE, market_ticker='URTH')
        
    except ValueError as e:
        print(f"Optimization failed: {e}. Check input data or constraints.")



if __name__ == "__main__":
    main()
    
    
    
    
    
    
