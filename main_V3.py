#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive Stock Portfolio Optimization Script
-------------------------------------------------
Created: Sat Nov 30 18:32:19 2024
Author: Adrien Picard

Description:
This script performs portfolio optimization for a set of stocks using GMVP 
(Global Minimum Variance Portfolio) and MSR (Maximum Sharpe Ratio) techniques. 
It provides exploratory data analysis, efficient frontier visualization, 
correlation heatmaps, advanced financial metrics (Fama-French 3-Factor modeling), 
and performance evaluation metrics like Jensen's Alpha.
"""

import logging
import yfinance as yf
import numpy as np
import pandas as pd
import json
from datetime import datetime
import scipy.optimize as sco
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from pandas_datareader import data as pdr
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning) #Use to avoid Warnings and Noisy printings from the algos
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(
    level=logging.INFO,  # Set the minimum severity level to INFO
    format='%(asctime)s - %(levelname)s - %(message)s',  # Customize log message format
    handlers=[
        logging.FileHandler("portfolio_optimization.log"),  # Log to a file
        logging.StreamHandler()  # Log to console (optional)
    ]
)

def load_config(config_file="config.json"):
    """
    Loads configuration parameters from a JSON file.

    Args:
        config_file (str, optional): Path to the JSON configuration file. Defaults to "config.json".

    Returns:
        dict: Dictionary containing configuration parameters.
    """
    try:
        with open(config_file, "r") as file:
            config = json.load(file)

        # Handle optional or dynamic values
        if config["end_date"] is None:
            config["end_date"] = datetime.today().strftime('%Y-%m-%d')

        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file '{config_file}' not found.")
    except json.JSONDecodeError as e:
        raise ValueError(f"Error parsing JSON configuration file: {e}")

config = load_config()

# Constants
TICKERS = config["tickers"]
START_DATE = config["start_date"]
END_DATE = config["end_date"]
RISK_FREE_RATE = config["risk_free_rate"]
PATH_TO_PLOTS = config["output_path"]
BASE_CURRENCY = config["base_currency"]

import sys
import subprocess

def install_required_packages():
    """
    Installs required Python packages for the script if they are not already installed.
    
    Raises:
        RuntimeError: If the Python version is lower than 3.11.
    """
    required_packages = ['yfinance', 'numpy', 'pandas', 'scipy', 'matplotlib', 'seaborn', 'statsmodels']
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    if sys.version_info < (3, 11):
        raise RuntimeError("This script requires Python 3.11 or newer")

install_required_packages()


# Utility Functions
def answer_input(prompt, default="yes"):
    """
    Prompts the user for input with a default value.

    Args:
        prompt (str): The question or message displayed to the user.
        default (str, optional): The default value returned if no input is provided. Defaults to "yes".

    Returns:
        str: The user's response or the default value.
    """
    print(prompt)
    return input().strip().lower() or default


def fetch_stock_data(ticker, start_date, end_date):
    """
    Fetches historical adjusted closing prices for a specific stock.

    Args:
        ticker (str): The stock's ticker symbol.
        start_date (str): The start date for fetching data in 'YYYY-MM-DD' format.
        end_date (str): The end date for fetching data in 'YYYY-MM-DD' format.

    Returns:
        pd.Series: Time series of adjusted closing prices.
        None: If no data is available or an error occurs.
    """
    try:
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)['Adj Close']
        return data if not data.empty else None
    except Exception as e:
        logging.error(f"Error fetching data for {ticker}: {e}")
        return None

def get_company_name(ticker):
    """
    Retrieves the company's short name for a given stock ticker.

    Args:
        ticker (str): The stock's ticker symbol.

    Returns:
        str: The company's short name, or the ticker if the name cannot be retrieved.
    """
    try:
        return yf.Ticker(ticker).info.get('shortName', ticker)
    except Exception:
        logging.error(f'Error running get_company_name for {ticker}')
        return ticker

def check_tickers_fetching(tickers, start_date, end_date):
    """
    Validates and fetches historical data for a list of stock tickers.

    Args:
        tickers (list of str): List of stock ticker symbols.
        start_date (str): The start date for fetching data in 'YYYY-MM-DD' format.
        end_date (str): The end date for fetching data in 'YYYY-MM-DD' format.

    Returns:
        tuple:
            - list of str: Valid stock tickers with available data.
            - dict: Dictionary mapping tickers to their time series data.
            - list of str: Company names corresponding to valid tickers.
            - pd.DataFrame: Combined DataFrame of all valid tickers' adjusted closing prices.

    Raises:
        ValueError: If no valid data is downloaded for the provided tickers.
    """
    valid_tickers = []
    data_dict = {}
    company_names = []

    for ticker in tickers:# First, collect all valid data
        data = fetch_stock_data(ticker, start_date, end_date)
        if data is not None:
            valid_tickers.append(ticker)
            data_dict[ticker] = data
            company_names.append(get_company_name(ticker))
    if not valid_tickers:
        raise ValueError("No valid data downloaded for the provided tickers.")
   
    data = data_dict[valid_tickers[0]] # Create DataFrame from the first valid ticker's data
    
    for ticker in valid_tickers[1:]:# Join other tickers' data
        data = pd.concat([data, data_dict[ticker]], axis=1)
   
    data.columns = valid_tickers # Set column names
    return valid_tickers, data_dict, company_names, data

def filter_extreme_values(metrics_df, SPREAD=1.8):
    """
    Filters out stocks with extreme annual returns or volatilities.

    Args:
        metrics_df (pd.DataFrame): DataFrame containing annual return and volatility metrics.
        SPREAD (float, optional): Multiplier for the mean to determine extreme values. Defaults to 1.8.

    Returns:
        pd.DataFrame: Filtered DataFrame with extreme values excluded.
    """
    return_threshold = metrics_df["Annual_Return%"].mean() * SPREAD
    volatility_threshold = metrics_df["Annual_Volatility%"].mean() * SPREAD
    
    is_extreme = (metrics_df["Annual_Return%"].abs() > return_threshold) | (metrics_df["Annual_Volatility%"] > volatility_threshold)
    
    extreme_stocks = metrics_df[is_extreme]
    for _, row in extreme_stocks.iterrows():
        print(f"\nTicker {row['Stock_Ticker']} has extreme values: Return={row['Annual_Return%']:.2f}%, Volatility={row['Annual_Volatility%']:.2f}%")
        if answer_input("Do you want to include this stock in the analysis? (yes/no): ") != "yes":
            print(f"\nExcluding {row['Stock_Ticker']} from the analysis.")
            metrics_df = metrics_df[metrics_df.Stock_Ticker != row['Stock_Ticker']]
    return metrics_df

def compute_stock_metrics(tickers, start_date, end_date):
    """
    Computes financial metrics such as annual return and volatility for a list of stocks.

    Args:
        tickers (list of str): List of stock ticker symbols.
        start_date (str): The start date for fetching data in 'YYYY-MM-DD' format.
        end_date (str): The end date for fetching data in 'YYYY-MM-DD' format.

    Returns:
        dict: A dictionary containing:
            - "Metrics_DataFrame" (pd.DataFrame): Annual return and volatility metrics.
            - "Correlation_Matrix" (pd.DataFrame): Correlation matrix of daily returns.
            - "Covariance_Matrix" (pd.DataFrame): Covariance matrix of daily returns.
            - "Raw Data" (pd.DataFrame): Adjusted closing prices of the valid stocks.
    """
    valid_tickers, data_dict, company_names, data = check_tickers_fetching(tickers, start_date, end_date)
    daily_returns = data.pct_change()
    
    metrics_df = pd.DataFrame({
        "Stock_Ticker": valid_tickers,
        "Company_Name": company_names,
        "Annual_Return%": daily_returns.mean() * 252 * 100,
        "Annual_Volatility%": daily_returns.std() * np.sqrt(252) * 100
    })
    
    metrics_df = filter_extreme_values(metrics_df) # Apply filter_extreme_values
    
    metrics_df = metrics_df.dropna() # Remove rows with any NaN values
    
    # Recalculate correlation and covariance matrices with clean data
    clean_daily_returns = daily_returns[metrics_df["Stock_Ticker"]]
    correlation_matrix = clean_daily_returns.corr()
    cov_matrix = clean_daily_returns.cov() * 252
    
    return {
        "Metrics_DataFrame": metrics_df,
        "Correlation_Matrix": correlation_matrix,
        "Covariance_Matrix": cov_matrix,
        "Raw Data": data[metrics_df["Stock_Ticker"]]
    }

# Function to print portfolio weights in a readable format
def print_portfolio_weights(weights, tickers):
    """
    Prints portfolio weights in a tabular format.

    Args:
        weights (list of float): Portfolio weights for each stock.
        tickers (list of str): Stock ticker symbols corresponding to the weights.
    """
    print("\nPortfolio Weights:")
    print(f"{'Ticker':<10}{'Weight (%)':<15}")
    print("-" * 25)
    for ticker, weight in zip(tickers, weights):
        print(f"{ticker:<10}{weight * 100:<15.2f}")
        
# Portfolio Optimization Functions
def portfolio_metrics(weights, returns, cov_matrix):
    """
    Calculates portfolio return, volatility, and Sharpe ratio.

    Args:
        weights (np.ndarray): Portfolio weights.
        returns (np.ndarray): Annualized returns for each stock.
        cov_matrix (np.ndarray): Covariance matrix of stock returns.

    Returns:
        tuple:
            - float: Portfolio return.
            - float: Portfolio volatility.
            - float: Sharpe ratio of the portfolio.
    """
    portfolio_return = np.dot(weights, returns)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe_ratio = (portfolio_return - RISK_FREE_RATE) / portfolio_volatility
    return portfolio_return, portfolio_volatility, sharpe_ratio

def optimize_portfolio(mean_returns, cov_matrix, opt_type):
    """
    Optimizes portfolio weights based on the chosen optimization type.

    Args:
        mean_returns (np.ndarray): Annualized mean returns for each stock.
        cov_matrix (np.ndarray): Covariance matrix of stock returns.
        opt_type (str): Optimization type ("GMVP" for Global Minimum Variance Portfolio or "MSR" for Maximum Sharpe Ratio).

    Returns:
        tuple:
            - np.ndarray: Optimized portfolio weights.
            - float: Objective function value (e.g., portfolio volatility for GMVP).
    """
    num_assets = len(mean_returns)
    constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}
    bounds = tuple((0.0, 1.0) for _ in range(num_assets))
    
    if opt_type == "GMVP":
        objective = lambda weights: np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    else:
        objective = lambda weights: -(np.dot(weights, mean_returns) - RISK_FREE_RATE) / np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    
    result = sco.minimize(objective, num_assets * [1.0 / num_assets], method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x, result.fun

def calculate_efficient_frontier(mean_returns, cov_matrix, num_points=100):
    """
    Computes the efficient frontier by varying target returns.

    Args:
        mean_returns (np.ndarray): Annualized mean returns for each stock.
        cov_matrix (np.ndarray): Covariance matrix of stock returns.
        num_points (int, optional): Number of points to compute on the efficient frontier. Defaults to 100.

    Returns:
        tuple:
            - np.ndarray: Target returns.
            - np.ndarray: Corresponding portfolio volatilities.
    """
    target_returns = np.linspace(mean_returns.min(), mean_returns.max(), num_points)
    efficient_volatilities = []
    
    for target_return in target_returns:
        constraints = (
            {"type": "eq", "fun": lambda weights: np.sum(weights) - 1},
            {"type": "eq", "fun": lambda weights: np.dot(weights, mean_returns) - target_return}
        )
        bounds = [(0.0, 1.0) for _ in range(len(mean_returns))]
        result = sco.minimize(
            lambda weights: np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))),
            np.ones(len(mean_returns)) / len(mean_returns),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        efficient_volatilities.append(result.fun * 100 if result.success else np.nan)
    
    target_returns = target_returns[~np.isnan(efficient_volatilities)]
    efficient_volatilities = np.array(efficient_volatilities)[~np.isnan(efficient_volatilities)]
    return target_returns * 100, efficient_volatilities


def simulate_portfolios(returns, cov_matrix, num_portfolios=2000):
    """
    Simulates random portfolios for Monte Carlo analysis.

    Args:
        returns (np.ndarray): Annualized mean returns for each stock.
        cov_matrix (np.ndarray): Covariance matrix of stock returns.
        num_portfolios (int, optional): Number of random portfolios to simulate. Defaults to 2000.

    Returns:
        tuple:
            - np.ndarray: Simulated portfolio returns.
            - np.ndarray: Simulated portfolio volatilities.
            - np.ndarray: Simulated Sharpe ratios.
    """
    num_assets = len(returns)
    weights = np.random.rand(num_portfolios, num_assets)
    weights /= np.sum(weights, axis=1)[:, np.newaxis]

    portfolio_returns = np.dot(weights, returns) * 100
    portfolio_volatilities = np.sqrt(np.einsum('ij,jk,ik->i', weights, cov_matrix, weights)) * 100
    sharpe_ratios = (portfolio_returns - RISK_FREE_RATE * 100) / portfolio_volatilities

    return portfolio_returns, portfolio_volatilities, sharpe_ratios


def create_plot(figsize=(12, 8), title="", xlabel="", ylabel="", dpi=300):
    """
    Creates a styled Matplotlib plot with basic configurations.

    Args:
        figsize (tuple, optional): Figure dimensions (width, height). Defaults to (12, 8).
        title (str, optional): Title of the plot. Defaults to "".
        xlabel (str, optional): Label for the X-axis. Defaults to "".
        ylabel (str, optional): Label for the Y-axis. Defaults to "".
        dpi (int, optional): Resolution of the plot. Defaults to 300.

    Returns:
        matplotlib.axes.Axes: The configured axes object.
    """
    plt.figure(figsize=figsize, dpi=dpi)
    sns.set_style("whitegrid")
    plt.title(title, fontsize=16, pad=20)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.grid(alpha=0.3)
    return plt.gca()

def finalize_plot(ax, filename, legend=True, legend_title=None, tight_layout=True, show=True):
    """
    Finalizes a Matplotlib plot by adding legends, saving to file, and optionally displaying it.

    Args:
        ax (matplotlib.axes.Axes): The plot's axes object.
        filename (str): Filename to save the plot.
        legend (bool, optional): Whether to include a legend. Defaults to True.
        legend_title (str, optional): Title for the legend. Defaults to None.
        tight_layout (bool, optional): Whether to apply a tight layout. Defaults to True.
        show (bool, optional): Whether to display the plot. Defaults to True.
    """
    if legend:
        if legend_title:
            ax.legend(title=legend_title, fontsize=10, loc="best", frameon=True, fancybox=True, framealpha=0.6)
        else:
            ax.legend(fontsize=10, loc="best", frameon=True, fancybox=True, framealpha=0.6)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(f'{PATH_TO_PLOTS}{filename}.png')
    plt.show()

def fetch_and_convert_data(tickers, start_date, end_date, base_currency="USD"):
    """
    Fetches historical stock prices and converts them to a specified base currency.

    Args:
        tickers (list of str): List of stock ticker symbols.
        start_date (str): Start date for fetching data in 'YYYY-MM-DD' format.
        end_date (str): End date for fetching data in 'YYYY-MM-DD' format.
        base_currency (str, optional): Target currency for conversion. Defaults to "USD".

    Returns:
        dict: A dictionary where keys are tickers and values are time series of converted prices.
    """
    data_dict = {}
    conversion_rates = {"USD": 1.0, "EUR": 1.1, "JPY": 0.008, "GBP": 1.25}
    
    for ticker in tickers:
        try:
            stock_info = yf.Ticker(ticker).info
            stock_currency = stock_info.get('currency', 'Unknown')
            
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)['Adj Close']
            
            if not data.empty and stock_currency in conversion_rates:
                data = data * conversion_rates[stock_currency]
                data_dict[ticker] = data
            else:
                print(f"No data or missing conversion rate for {ticker}. Skipping...")
        except Exception as e:
            logging.error(f"Error processing {ticker}: {e}") 
    return data_dict


# Visualization Functions
def plot_historical_prices(metrics_df, start_date, end_date, base_currency="USD", title="Historical Prices (Converted)"):
    """
    Plots historical stock prices converted to a specified currency.

    Args:
        metrics_df (pd.DataFrame): DataFrame containing stock tickers and company names.
        start_date (str): Start date for fetching data in 'YYYY-MM-DD' format.
        end_date (str): End date for fetching data in 'YYYY-MM-DD' format.
        base_currency (str, optional): Target currency for price conversion. Defaults to "USD".
        title (str, optional): Title of the plot. Defaults to "Historical Prices (Converted)".
    """
    ax = create_plot(figsize=(14, 8), title=title, xlabel="Date", ylabel=f"Adjusted Price ({base_currency})")
    
    tickers = metrics_df['Stock_Ticker'].tolist()
    company_names = metrics_df['Company_Name'].tolist()
    data_dict = fetch_and_convert_data(tickers, start_date, end_date, base_currency)
    
    for ticker, name in zip(tickers, company_names):
        if ticker in data_dict:
            ax.plot(data_dict[ticker].index, data_dict[ticker], label=f"{name} ({ticker})", linewidth=2)
    
    plt.figtext(0.5, -0.05, f"Note: Prices are converted to {base_currency} using static exchange rates; stocks with no data are skipped.",
                wrap=True, horizontalalignment='center', fontsize=10, color="gray")
    
    finalize_plot(ax, "Historical_prices", legend_title="Stocks")

def plot_correlation_heatmap(correlation_matrix, tickers, title="Correlation Heatmap"):
    """
    Plots a heatmap of the correlation matrix for the given stocks.

    Args:
        correlation_matrix (pd.DataFrame): Correlation matrix of stock returns.
        tickers (list of str): Stock ticker symbols.
        title (str, optional): Title of the plot. Defaults to "Correlation Heatmap".
    """
    ax = create_plot(figsize=(10, 8), title=title)
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f",
                xticklabels=tickers, yticklabels=tickers, linewidths=0.5,
                cbar_kws={'label': 'Correlation Coefficient'}, ax=ax)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    finalize_plot(ax, "Correlation_heatmap", legend=False)

def plot_gmvp_msr_weights(tickers, company_names, gmvp_weights, msr_weights, gmvp_performance, msr_performance):
    """
    Visualizes the portfolio weights for GMVP and MSR optimization.

    Args:
        tickers (list of str): List of stock ticker symbols.
        company_names (list of str): List of corresponding company names.
        gmvp_weights (np.ndarray): Optimized weights for the GMVP portfolio.
        msr_weights (np.ndarray): Optimized weights for the MSR portfolio.
        gmvp_performance (tuple): Performance metrics for GMVP (return, volatility).
        msr_performance (tuple): Performance metrics for MSR (return, volatility).
    """
    ax = create_plot(figsize=(12, 8), title="Portfolio Weights for GMVP and MSR", xlabel="Stocks", ylabel="Weight (%)")
    
    labels = [f"{name} ({ticker})" for name, ticker in zip(company_names, tickers)]
    x = np.arange(len(labels))
    width = 0.1
    offset = 0.15
    
    bars_gmvp = ax.bar(x - offset, gmvp_weights * 100, width, label='GMVP Weights', color='red', edgecolor='black')
    bars_msr = ax.bar(x + offset, msr_weights * 100, width, label='MSR Weights', color='blue', edgecolor='black')
    
    for bars, weights in [(bars_gmvp, gmvp_weights), (bars_msr, msr_weights)]:
        for bar, weight in zip(bars, weights):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"{weight * 100:.1f}%", ha='center', va='bottom', fontsize=10, color='black')
    
    gmvp_return, gmvp_volatility = gmvp_performance
    msr_return, msr_volatility = msr_performance
    
    ax.text(len(labels) - 1, max(max(gmvp_weights), max(msr_weights)) * 100 - 15,
            f"GMVP:\nReturn={gmvp_return * 100:.2f}%,\nVolatility={gmvp_volatility * 100:.2f}%",
            color="red", fontsize=8, ha='center',
            bbox=dict(facecolor="white", alpha=0.4, edgecolor="gray"))
    
    ax.text(len(labels) - 1, max(max(gmvp_weights), max(msr_weights)) * 100 - 25,
            f"MSR:\nReturn={msr_return * 100:.2f}%,\nVolatility={msr_volatility * 100:.2f}%",
            color="blue", fontsize=8, ha='center',
            bbox=dict(facecolor="white", alpha=0.4, edgecolor="gray"))
    
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10, rotation=45, ha='right')
    finalize_plot(ax,"Gmvp_msr_weights")

def plot_return_vs_risk_portfolio_GMVP_MSR(stock_df, simulated_returns, simulated_volatilities, simulated_sharpe_ratios,
                                           efficient_returns, efficient_volatilities, msr_return, msr_volatility,
                                           min_risk_return, min_risk_volatility):
    """
    Plots the risk-return characteristics of the portfolio and individual assets.

    Args:
        stock_df (pd.DataFrame): DataFrame containing stock return and volatility metrics.
        simulated_returns (np.ndarray): Simulated portfolio returns.
        simulated_volatilities (np.ndarray): Simulated portfolio volatilities.
        simulated_sharpe_ratios (np.ndarray): Simulated Sharpe ratios.
        efficient_returns (np.ndarray): Returns on the efficient frontier.
        efficient_volatilities (np.ndarray): Volatilities on the efficient frontier.
        msr_return (float): Return of the Maximum Sharpe Ratio portfolio.
        msr_volatility (float): Volatility of the Maximum Sharpe Ratio portfolio.
        min_risk_return (float): Return of the Global Minimum Variance Portfolio.
        min_risk_volatility (float): Volatility of the Global Minimum Variance Portfolio.
    """
    ax = create_plot(figsize=(14, 10), title="Expected Return vs Risk for Portfolio and Individual Assets",
                     xlabel="Risk (Volatility %)", ylabel="Expected Return (%)")
    
    scatter = ax.scatter(simulated_volatilities, simulated_returns, c=simulated_sharpe_ratios, cmap="viridis", alpha=0.6, s=10)
    plt.colorbar(scatter, label="Sharpe Ratio", shrink=1, aspect=30, pad=0.02)
    
    ax.scatter(0, RISK_FREE_RATE * 100, color="lime", label="Risk-Free Rate", edgecolor="Teal", s=150, zorder=3)
    ax.text(-0.2, RISK_FREE_RATE * 100 + 8, "Risk-Free\n(5%)", fontsize=10, ha="center",
            bbox=dict(facecolor="white", edgecolor="gray", boxstyle="round,pad=0.3"))
    
    for _, row in stock_df.iterrows():
        ax.scatter(row["Stock_volatility%"], row["Stock_return%"], color="orange", edgecolor="black", s=100, zorder=3)
        ax.text(row["Stock_volatility%"] + 0.5, row["Stock_return%"] + 2,
                f"{row['Stock_name']}\n({row['Stock_return%']:.1f}%)", fontsize=9,
                bbox=dict(facecolor="white", edgecolor="gray", boxstyle="round,pad=0.3"))
        ax.plot([row["Stock_volatility%"], min_risk_volatility * 100],
                [row["Stock_return%"], min_risk_return * 100], linestyle="dotted", color="gray")
    
    ax.scatter(min_risk_volatility * 100, min_risk_return * 100, color="red", edgecolor="gray",
               label="GMVP (Minimum Risk)", marker="H", s=180, zorder=3)
    ax.text(min_risk_volatility * 100 - 3.2, min_risk_return * 100 + 2,
            f"GMVP\n({min_risk_return * 100:.2f}%)", fontsize=10,
            bbox=dict(facecolor="white", edgecolor="gray", boxstyle="round,pad=0.3"))
    
    ax.scatter(msr_volatility * 100, msr_return * 100, color="blue",
               label="MSR (Maximum Sharpe Ratio) Portfolio", marker="o", s=180, zorder=3)
    ax.text(msr_volatility * 100 - 3, msr_return * 100 + 5,
            f"MSR\n({msr_return * 100:.2f}%)", fontsize=10,
            bbox=dict(facecolor="white", edgecolor="gray", boxstyle="round,pad=0.3"))
    
    ax.plot(efficient_volatilities, efficient_returns, color="blue", linestyle="--", linewidth=1.5, label="Efficient Frontier")
    
    ax.set_xticks(np.arange(0, max(efficient_volatilities) + 1, 2))
    ax.set_yticks(np.arange(0, max(efficient_returns) + 1, 5))
    
    cml_x = np.linspace(0, msr_volatility * 100, 100)
    cml_y = RISK_FREE_RATE * 100 + (msr_return * 100 - RISK_FREE_RATE * 100) * (cml_x / (msr_volatility * 100))
    ax.plot(cml_x, cml_y, label="Capital Market Line (CML)", color="green", linestyle="-", linewidth=1.5)
    
    finalize_plot(ax, "Return_vs_risk_portfolio_GMVP_MSR")

def fetch_fama_french_factors(start_date, end_date):
    """
    Fetches Fama-French 3-factor data for the specified date range.

    Args:
        start_date (str): Start date for the data in 'YYYY-MM-DD' format.
        end_date (str): End date for the data in 'YYYY-MM-DD' format.

    Returns:
        pd.DataFrame: Daily Fama-French factors (MKT-RF, SMB, HML, RF).
    """
    ff3_data = pdr.DataReader('F-F_Research_Data_Factors_daily', 'famafrench', start_date, end_date)[0]
    ff3_data.index = pd.to_datetime(ff3_data.index)
    ff3_data = ff3_data.rename(columns={'Mkt-RF': 'MKT-RF', 'HML': 'HML', 'SMB': 'SMB', 'RF': 'RF'})
    ff3_data = ff3_data / 100
    return ff3_data

def compute_fama_french_3_factors(portfolio_returns, ff3_factors, portfolio_name="Portfolio"):
    """
    Performs Fama-French 3-factor regression on the portfolio's excess returns.

    Args:
        portfolio_returns (pd.Series): Portfolio returns over time.
        ff3_factors (pd.DataFrame): Fama-French 3-factor data.
        portfolio_name (str, optional): Name of the portfolio. Defaults to "Portfolio".

    Outputs:
        Plots of regression coefficients and fitted vs. actual returns.
    """
    common_dates = portfolio_returns.index.intersection(ff3_factors.index)
    portfolio_returns = portfolio_returns.loc[common_dates]
    ff3_factors = ff3_factors.loc[common_dates]
    portfolio_excess_returns = portfolio_returns - ff3_factors['RF']
    
    X = sm.add_constant(ff3_factors[['MKT-RF', 'SMB', 'HML']])

    model = sm.OLS(portfolio_excess_returns, X).fit()
    #print(model.summary()) #withdraw the '#' before the print if you want to access to OLS Regression Results

    factors = ['Alpha (Intercept)', 'MKT-RF', 'SMB', 'HML']
    coefficients = model.params
    
    ax = create_plot(figsize=(10, 6), title=f"Fama-French 3-Factor Coefficients for {portfolio_name}", 
                     xlabel="Factors", ylabel="Coefficient Value")
    ax.bar(factors, coefficients, color=['blue', 'orange', 'green', 'red'], edgecolor='black')
    finalize_plot(ax, "fama_french_3_fig1", legend=False)

    ax = create_plot(figsize=(10, 6), title=f"Fitted vs Actual Excess Returns for {portfolio_name}", 
                     xlabel="Actual Excess Returns", ylabel="Fitted Excess Returns")
    ax.scatter(portfolio_excess_returns, model.fittedvalues, alpha=0.7, label="Fitted vs Actual")
    ax.plot([portfolio_excess_returns.min(), portfolio_excess_returns.max()], 
            [portfolio_excess_returns.min(), portfolio_excess_returns.max()], 
            color="red", linestyle="--", label="45° Line")
    finalize_plot(ax, "fama_french_3_fig2")

    print("\n\n=== Fama-French Factor Ratios ===\n")
    factor_stats = pd.DataFrame({
        'Mean': ff3_factors.mean(),
        'Std Dev': ff3_factors.std(),
        'Correlation with Portfolio Excess Returns': ff3_factors.corrwith(portfolio_excess_returns)
    }).loc[['MKT-RF', 'SMB', 'HML', 'RF']]
    print(factor_stats)

    ax = create_plot(figsize=(10, 6), title="Fama-French Factor Mean and Standard Deviation", 
                     xlabel="Factors", ylabel="Value")
    factor_stats[['Mean', 'Std Dev']].plot(kind='bar', ax=ax, edgecolor='black')
    finalize_plot(ax, "fama_french_3_fig3")

def plot_sml_with_dynamic_gmvp(metrics_df, gmvp_weights, market_ticker='URTH', start_date=START_DATE, end_date=END_DATE):
    """
    Plots the Security Market Line (SML) with the GMVP's beta and return.

    Args:
        metrics_df (pd.DataFrame): DataFrame containing stock metrics (beta, return, volatility).
        gmvp_weights (np.ndarray): Weights for the GMVP portfolio.
        market_ticker (str, optional): Ticker symbol for the market index. Defaults to 'URTH'.
        start_date (str): Start date for fetching data in 'YYYY-MM-DD' format.
        end_date (str): End date for fetching data in 'YYYY-MM-DD' format.
    """
    try:
        # Download and prepare market data
        market_data = yf.download(market_ticker, start=start_date, end=end_date, progress=False)        
        if market_data.empty:
            raise ValueError("No data found for market ticker.")
        
        market_daily_returns = market_data['Adj Close'].pct_change().dropna()
        market_annual_return = ((1 + market_daily_returns.mean()) ** 252) - 1

        # Calculate betas for stocks
        stock_betas = []
        for ticker in metrics_df['Stock_Ticker']:
            stock_data = yf.download(ticker, start=start_date, end=end_date, progress=False)['Adj Close'].pct_change().dropna()
            
            common_dates = stock_data.index.intersection(market_daily_returns.index)
            stock_data = stock_data.loc[common_dates]
            market_returns = market_daily_returns.loc[common_dates]
            
            X = sm.add_constant(market_returns)
            model = sm.OLS(stock_data, X).fit()
            stock_betas.append(model.params[1])
        
        metrics_df['Beta'] = stock_betas
        gmvp_beta = np.dot(gmvp_weights, metrics_df['Beta'])
        gmvp_return = np.dot(gmvp_weights, metrics_df['Annual_Return%']) / 100

        # Extract the scalar value from market_annual_return
        market_annual_return_value = market_annual_return.iloc[0]

        # Prepare data for SML
        betas = np.linspace(-0.5, 2.5, 100)
        sml_returns = RISK_FREE_RATE + betas * (market_annual_return_value - RISK_FREE_RATE)

        # Plot SML
        ax = create_plot(title="Security Market Line (SML) with GMVP", xlabel="Beta (Systematic Risk)", ylabel="Expected Return (%)")
        ax.plot(betas, sml_returns * 100, label="SML", linestyle="--", color="blue")

        # Plot GMVP point with label
        ax.scatter(gmvp_beta, gmvp_return * 100, color="red", s=100, label="GMVP")
        ax.annotate(f"GMVP ({gmvp_return:.2%})", 
                    (gmvp_beta, gmvp_return * 100), 
                    xytext=(5, 5), 
                    textcoords='offset points', 
                    color='red', 
                    fontsize=8)

        # Plot individual stocks with labels
        for _, row in metrics_df.iterrows():
            ax.scatter(row['Beta'], row['Annual_Return%'], color='orange', s=50)
            ax.annotate(row['Stock_Ticker'], 
                        (row['Beta'], row['Annual_Return%']), 
                        xytext=(5, 5), 
                        textcoords='offset points', 
                        color='black', 
                        fontsize=8)

        ax.axhline(y=RISK_FREE_RATE * 100, color="green", linestyle="-", label="Risk-Free Rate")
        ax.scatter([], [], color='orange', s=50, label='Stocks')# Add a single point for stocks in the legend
        ax.legend(loc='best', fontsize=8)   # Adjust legend
        finalize_plot(ax, "sml_with_dynamic_gmvp")
    except Exception as e:
        logging.error(f"Error in plot_sml_with_dynamic_gmvp: {e}")

def evaluate_gmvp_performance(gmvp_weights, gmvp_return, gmvp_volatility, cov_matrix, risk_free_rate, market_ticker, start_date, end_date, data, tickers):
    """
    Evaluates the performance of the GMVP portfolio using various metrics.

    Args:
        gmvp_weights (np.ndarray): Weights of the GMVP portfolio.
        gmvp_return (float): Return of the GMVP portfolio.
        gmvp_volatility (float): Volatility of the GMVP portfolio.
        cov_matrix (np.ndarray): Covariance matrix of stock returns.
        risk_free_rate (float): Risk-free rate used for calculations.
        market_ticker (str): Ticker symbol for the market index.
        start_date (str): Start date for fetching data in 'YYYY-MM-DD' format.
        end_date (str): End date for fetching data in 'YYYY-MM-DD' format.
        data (pd.DataFrame): Raw data of stock prices.
        tickers (list of str): List of stock tickers in the portfolio.

    Returns:
        dict: Performance metrics including Sharpe ratio, beta, Jensen's alpha, and more.
    """
    try:
        market_data = yf.download(market_ticker, start=start_date, end=end_date, progress=False)["Adj Close"]
        if market_data.empty:
            return {"Error": "Market data is empty."}

        market_log_returns = np.log(market_data / market_data.shift(1)).dropna()
        log_returns = np.log(data / data.shift(1)).dropna()

        # Ensure one-dimensionality
        if len(log_returns.shape) > 1:
            log_returns = log_returns.squeeze()
        if len(market_log_returns.shape) > 1:
            market_log_returns = market_log_returns.squeeze()

        aligned_data = log_returns.join(market_log_returns.rename("Market"), how="inner").dropna()

        market_variance = aligned_data["Market"].var()
        if market_variance == 0 or np.isnan(market_variance):
            return {"Error": "Market returns variance is zero or invalid."}

        # Calculate portfolio beta
        betas = aligned_data.cov().loc[tickers, "Market"] / market_variance
        portfolio_beta = np.dot(gmvp_weights, betas)
       
        annualized_market_return = ((1 + market_log_returns.mean()) ** 252) - 1

        annualized_market_volatility = market_log_returns.std() * np.sqrt(252)
        jensens_alpha = gmvp_return - (risk_free_rate + portfolio_beta * (annualized_market_return - risk_free_rate))

        # Compile performance metrics
        performance_metrics = {
            "GMVP Annualized Return (%)": gmvp_return * 100,
            "GMVP Annualized Volatility (%)": gmvp_volatility * 100,
            "GMVP Sharpe Ratio": (gmvp_return - risk_free_rate) / gmvp_volatility,
            "GMVP Beta": portfolio_beta,
            "GMVP Jensen's Alpha": jensens_alpha,
            "Market Annualized Return (%)": annualized_market_return * 100,
            "Market Annualized Volatility (%)": annualized_market_volatility * 100
        }

        return performance_metrics
    except Exception as e:
        logging.error(f"Error in performance evaluation: {e}")
        return {}

def main():
    """
    Executes the end-to-end portfolio optimization and analysis workflow:
    - Computes stock metrics.
    - Optimizes portfolios for GMVP and MSR.
    - Evaluates performance and generates plots.
    - Prints key results and metrics.

    Outputs:
        Results and visualizations in the console and saved files.
    """
    try:
        # Step 1: Compute stock metrics
        stock_metrics = compute_stock_metrics(TICKERS, START_DATE, END_DATE)
        metrics_df = stock_metrics["Metrics_DataFrame"]
        correlation_matrix = stock_metrics["Correlation_Matrix"]
        cov_matrix = stock_metrics["Covariance_Matrix"]
        data = stock_metrics['Raw Data']

        if metrics_df.empty:
            print("No valid stocks remaining after filtering.")
            return

        # Extract necessary values
        returns = metrics_df["Annual_Return%"].values / 100
        tickers = metrics_df["Stock_Ticker"].values

        # Step 2: Portfolio Optimization
        gmvp_weights, gmvp_volatility = optimize_portfolio(returns, cov_matrix, "GMVP")
        gmvp_return = np.dot(gmvp_weights, returns)

        msr_weights, _ = optimize_portfolio(returns, cov_matrix, "MSR")
        msr_return, msr_volatility, _ = portfolio_metrics(msr_weights, returns, cov_matrix)

        # Print Basic Portfolio Results
        print("\n\n===== NEOMA WORLD GROWTH PORTFOLIO =====")
        print(f"\n*Data period: {START_DATE} to {END_DATE}\n")
        print("\nTicker    Return (%)     Volatility (%)    Company Name")
        print("-" * 70)
        for ticker, ret, vol, name in zip(
            tickers, returns * 100, metrics_df['Annual_Volatility%'], metrics_df['Company_Name']
        ):
            print(f"{ticker:<10}{ret:<15.2f}{vol:<20.2f}{name:<23}")

        # Print Optimization Results
        print("\n\n=== Portfolio Metrics ===")
        print(f"\nGMVP Return: {gmvp_return * 100:.2f}%, GMVP Volatility: {gmvp_volatility * 100:.2f}%")
        print_portfolio_weights(gmvp_weights, tickers)
        print(f"\n\nMSR Return: {msr_return * 100:.2f}%, MSR Volatility: {msr_volatility * 100:.2f}%")
        print_portfolio_weights(msr_weights, tickers)

        # Step 3: Performance Evaluation
        performance_metrics = evaluate_gmvp_performance(
            gmvp_weights=gmvp_weights,
            gmvp_return=gmvp_return,
            gmvp_volatility=gmvp_volatility,
            cov_matrix=cov_matrix,
            risk_free_rate=RISK_FREE_RATE,
            market_ticker="URTH",
            start_date=START_DATE,
            end_date=END_DATE,
            data=data,
            tickers=tickers,
        )

        print("\n=== GMVP Performance Metrics ===\n")
        for key, value in performance_metrics.items():
            print(f"{key}: {value:.4f}" if isinstance(value, (float, int)) else f"{key}: {value}")

        # Step 4: Generate Plots
        plot_correlation_heatmap(correlation_matrix, tickers, title="Correlation Heatmap for Selected Stocks")

        efficient_returns, efficient_volatilities = calculate_efficient_frontier(returns, cov_matrix)
        simulated_returns, simulated_volatilities, simulated_sharpe_ratios = simulate_portfolios(returns, cov_matrix)

        stock_df = pd.DataFrame({
            "Stock_name": metrics_df['Stock_Ticker'],
            "Stock_return%": metrics_df['Annual_Return%'],
            "Stock_volatility%": metrics_df["Annual_Volatility%"]
        })

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
            gmvp_volatility,
        )

        plot_gmvp_msr_weights(
            tickers,
            metrics_df['Company_Name'],
            gmvp_weights,
            msr_weights,
            (gmvp_return, gmvp_volatility),
            (msr_return, msr_volatility),
        )

        plot_historical_prices(metrics_df, START_DATE, END_DATE, base_currency="USD", title="Historical Prices in USD")

        # Step 5: Fama-French Analysis
        portfolio_daily_returns = metrics_df['Annual_Return%'] / 252
        portfolio_returns = pd.Series(
            portfolio_daily_returns.values, index=pd.date_range(start=START_DATE, periods=len(portfolio_daily_returns), freq='B')
        )

        ff3_factors = fetch_fama_french_factors(START_DATE, END_DATE)
        compute_fama_french_3_factors(portfolio_returns, ff3_factors, portfolio_name="Sample Portfolio")

        plot_sml_with_dynamic_gmvp(metrics_df, gmvp_weights, start_date=START_DATE, end_date=END_DATE, market_ticker='URTH')

        print(f"\n - Project NEOMA Business School - MSc International Finance, FMRM Track - \n")
    except Exception as e:
        logging.error(f"An error occurred during execution: {e}")

if __name__ == "__main__":
    main()
