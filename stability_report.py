import datetime
import glob
import math
import sys

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import copy
import seaborn as sns
import stability_pool_simple
from joblib import Parallel, delayed
import multiprocessing
import uuid
import shutil
import os

sns.set_theme()


def get_volume_for_slippage(v, s, s1):
    x = (s * v) / (1 - s)
    # Slippage(qty) = X / (X + qty)
    # s1 = x / (x + v1)
    # s1 * (x + v1) = x
    # s1 * x + s1 * v1 = x
    # s1 * v1 = x - s1 * x
    # s1 * v1 = x * (1 - s1)
    v1 = (x * (1 - s1)) / s1
    return v1


def get_slippage_for_volume(v, s, v1):
    # x = y
    # x * y = k
    # (x + qty) * (x - v2) = k
    # (x - v2) = (x * x) / (x + qty)
    # (x - v2) / x = x / (x + qty)
    # Slippage(qty) = X / (X + qty)

    # s = x / (x + v)
    # s * (x + v) = x
    # s * x + s * v = x
    # s * v = x - s * x
    # s * v = x * (1 -s)
    x = (s * v) / (1 - s)

    # x' = x + v1
    # y' = x - ?
    # x'y' = x * x
    # priceimpact = y'/x'

    # (x + v) * (x - v1) = x * x
    # (x - v1) = (x * x) / (x + v)
    pi = (x * x) / pow(x + v1, 2)

    return pi


def plot_for_html(name):
    if os.path.isdir('results\\' + name):
        shutil.rmtree('results\\' + name)
    os.makedirs('results\\' + name)

    files = glob.glob("results\\*_" + name + "_*")
    all_df = pd.DataFrame()
    for file in files:
        df = pd.read_csv(file)
        if len(df) == 0:
            all_df = copy.deepcopy(df)
        else:
            all_df = all_df.append(df)

    all_df["Debt ceiling (M)"] = all_df["collateral"] / 1_000_000
    all_df["Monthly liquidation volume factor"] = all_df["collateral_liquidation_factor"]

    gg = ['price_recovery_time', 'volume_for_slippage_10_percents',
          'recovery_halflife_retail', 'stability_pool_initial_balance_ratio', 'share_institutional']

    uniques = all_df.groupby(gg).size().reset_index().rename(columns={0: 'count'})
    for index, row in uniques.iterrows():
        batch_df = copy.deepcopy(all_df)
        for g in gg:
            batch_df = batch_df.loc[batch_df[g] == row[g]]

        sns.set(font_scale=1.5)
        hm = get_heatmap(batch_df, "Debt ceiling (M)", "Monthly liquidation volume factor", "max_drop")
        ax = sns.heatmap(hm, annot=True, linewidths=.5, cmap="PiYG", vmin=0.5, vmax=1)
        file_name = "hm"
        for g in gg:
            file_name += "-" + g + "-" + str(row[g])

        fig = plt.gcf()
        fig.set_size_inches(12.5, 8.5)
        plt.savefig("results\\" + name + "\\" + file_name + ".jpg")
        plt.cla()
        plt.close()

    for index, row in all_df.iterrows():
        file_name = row["simulation_name"]
        df = pd.read_csv("results\\" + file_name + ".csv")
        fig, ax1 = plt.subplots()
        fig.set_size_inches(12.5, 8.5)
        ax2 = ax1.twinx()
        ax1.plot(df["ts"], df["price"], 'g-')
        ax2.plot(df["ts"], df["market_volume"] * ETH_PRICE / 1_000_000, 'r-', label="Market Volume")
        ax2.plot(df["ts"], df["stability_pool_available_volume"] * ETH_PRICE / 1_000_000, 'm-',
                 label="Stability Pool Liquidity")
        ax2.plot(df["ts"], df["open_liquidations"] * ETH_PRICE / 1_000_000, 'y-', label="Open Liquidations")
        ax2.plot(df["ts"], df["liquidation_volume"].rolling(30).sum() * ETH_PRICE / 1_000_000, 'b-',
                 label="30 minutes Liquidation Volume")

        ax1.set_xlabel('Time')
        ax1.set_ylabel('Price', color='g')

        gg = ['price_recovery_time', 'volume_for_slippage_10_percents', 'recovery_halflife_retail',
              'stability_pool_initial_balance_ratio', 'share_institutional', 'collateral',
              'collateral_liquidation_factor']

        file_name = "ts"
        for g in gg:
            file_name += "-" + g + "-" + str(row[g])
        plt.title("Max Drop:" + str(round(row["max_drop"], 2)))
        plt.legend()
        plt.savefig("results\\" + name + "\\" + file_name + ".jpg")
        plt.cla()
        plt.close()


def get_heatmap(df, x, y, z):
    df = copy.deepcopy(df)
    df[x] = df[x].astype(float)
    df[y] = df[y].astype(float)
    df[z] = df[z].astype(float)
    df["score"] = 1 - df[z]
    result = df.groupby([x, y])["score"].min().unstack(level=0)
    xx = df[y].unique()
    result.index = pd.CategoricalIndex(result.index, categories=sorted(xx, reverse=True))
    result.sort_index(inplace=True)
    return result


def calc_series_std_ratio(file_name, dai_eth, convert):
    print(file_name, dai_eth, convert)
    dai_eth = pd.read_csv(dai_eth)
    dai_eth["price"] = 1 / dai_eth["price"]

    test_eth = pd.read_csv(file_name)
    test_eth["price"] = 1 / test_eth["price"]

    test_eth = test_eth.loc[test_eth["qty1"] != 0]
    test_eth["price"] = test_eth["price"].astype(float)

    if convert:
        test_eth = test_eth.merge(dai_eth, how='inner', left_on=['block_number'], right_on=['block_number'])
        test_eth["price"] = test_eth["price_y"] / test_eth["price_x"]
        test_eth.to_csv("xxx.csv")

    dai_rolling_std = np.average(
        dai_eth["price"].rolling(5 * 30).std().dropna() / dai_eth["price"].rolling(5 * 30).mean().dropna())
    test_rolling_std = np.average(
        test_eth["price"].rolling(5 * 30).std().dropna() / test_eth["price"].rolling(5 * 30).mean().dropna())

    print("dai_avg", np.average(dai_eth["price"]))
    print("dai_min", np.min(dai_eth["price"]))
    print("dai_std", np.std(dai_eth["price"]) / np.average(dai_eth["price"]))

    print("test_avg", np.average(test_eth["price"]))
    print("test_min", np.min(test_eth["price"]))
    print("test_std", np.std(test_eth["price"]) / np.average(test_eth["price"]))

    print("30M Rolling STD Ratio", test_rolling_std / dai_rolling_std)

    return test_rolling_std / dai_rolling_std


def adjust_series_price(df, factor):
    last_price = 0
    last_adjusted_price = 0
    for index, row in df.iterrows():
        price = (row["ask_price"] + row["bid_price"]) * 0.5
        if last_price != 0:
            price_change = ((price / last_price) - 1) * factor
            adjust_price = last_adjusted_price + last_adjusted_price * price_change
        else:
            adjust_price = price

        df.at[index, "price"] = price
        df.at[index, "adjust_price"] = adjust_price
        last_adjusted_price = adjust_price
        last_price = price
    return copy.deepcopy(df)


def check_max_liquidation_drop(open_liquidations, price):
    m_drop = 0
    m_drop_volume = 0
    for open_liquidation in open_liquidations:
        if open_liquidation["closed"] != open_liquidation["liquidation_volume"]:
            drop = ((open_liquidation["price"] - price) / open_liquidation["price"])
            if m_drop < drop:
                m_drop = drop
                m_drop_volume = open_liquidation["liquidation_volume"] - open_liquidation["closed"]
    return m_drop, m_drop_volume


def convert_to_array(dai_eth):
    arr = []
    for index, row in dai_eth.iterrows():
        arr.append({
            "timestamp_x": row["timestamp_x"],
            "adjust_price": row["adjust_price"],
            liquidation_side: row[liquidation_side]})
    return arr


def run_simulation(file_name, name, config):
    if os.path.isfile("results\\" + file_name.replace("\\", "_") + "_" + name + "_" + "stability_report.csv"):
        print("Skipping", file_name, name)
        return ""
    series_std_ratio = config["series_std_ratio"]
    trade_every = config["trade_every"]
    report = []
    print(file_name)
    dai_eth = pd.read_csv(file_name)
    dai_eth = adjust_series_price(dai_eth, series_std_ratio)
    x = min(dai_eth["timestamp_x"])
    dai_eth_array = convert_to_array(dai_eth)
    file_description = datetime.datetime.fromtimestamp(x / (1000 * 1000))
    file_description = datetime.date.strftime(file_description, "%d/%m/%Y")
    file_total_volume = sum(dai_eth[liquidation_side])
    print(file_name, file_description, file_total_volume)
    simulation_id = str(uuid.uuid4())
    simulation_index = 0

    for volume_for_slippage_10_percents in config["volume_for_slippage_10_percentss"]:
        for l_factor in config["l_factors"]:
            for price_recovery_time in config["price_recovery_times"]:
                for share_institutional in config["share_institutionals"]:
                    for recovery_halflife_retail in config["recovery_halflife_retails"]:
                        for collateral in config["collaterals"]:
                            for l_incentive in config["liquidation_incentives"]:
                                for s_balance in config["stability_pool_initial_balances"]:
                                    simulation_index += 1
                                    simulation_name = str(simulation_id) + "_" + str(simulation_index)
                                    target_volume = collateral * l_factor
                                    liquidation_ratio = target_volume / file_total_volume
                                    cycle_trade_volume = get_volume_for_slippage(
                                        volume_for_slippage_10_percents * liquidation_factor, 1 - 0.1,
                                        1 - l_incentive)

                                    stability_pool_initial_balance = collateral * s_balance
                                    stability_pool_simple_instance = stability_pool_simple.stability_pool(
                                        initial_balance=stability_pool_initial_balance,
                                        recovery_interval=trade_every, recovery_volume=cycle_trade_volume,
                                        share_institutional=share_institutional,
                                        recovery_halflife_retail=recovery_halflife_retail)

                                    print(" " * 5,
                                          "file_name", file_name, "collateral", collateral, "l_factor", l_factor,
                                          "price_recovery_time", price_recovery_time, "l_incentive", l_incentive,
                                          "cycle_trade_volume", cycle_trade_volume,
                                          "s_balance", s_balance, "stability_pool_initial_balance",
                                          stability_pool_initial_balance)

                                    ts_report = []
                                    price_liquidation_factor = 1
                                    historical_cycle_trade_volume = []
                                    closed_liquidations = []
                                    open_liquidations = []
                                    max_drop = 0
                                    max_drop_open_volume = 0
                                    price_at_max_drop = 0
                                    max_liquidation_volume = 0
                                    min_price_liquidation_factor = float('inf')
                                    min_multiply_price_liquidation_factor = float('inf')
                                    all_liquidations_volume = 0
                                    max_daily_volume = 0
                                    last_row_date = 0
                                    daily_volume = 0
                                    for row in dai_eth_array:
                                        time = row["timestamp_x"]
                                        row_liquidation = row[liquidation_side]
                                        liquidation_volume = (row_liquidation * liquidation_ratio) / liquidation_factor
                                        max_liquidation_volume = max(liquidation_volume, max_liquidation_volume)

                                        row_date = datetime.datetime.fromtimestamp(time / (1000 * 1000))
                                        row_date = datetime.date.strftime(row_date, "%d/%m/%Y")
                                        if row_date != last_row_date:
                                            last_row_date = row_date
                                            if max_daily_volume < daily_volume:
                                                max_daily_volume = daily_volume
                                            daily_volume = 0

                                        daily_volume += liquidation_volume

                                        if row_liquidation > 0 and liquidation_volume == 0:
                                            print("row_liquidation", row_liquidation, liquidation_volume,
                                                  liquidation_ratio, target_volume / file_total_volume)
                                            exit()

                                        all_liquidations_volume += liquidation_volume

                                        # recover price_liquidation_factor
                                        missing_price_liquidation_factor = 1 - price_liquidation_factor
                                        if price_recovery_time == 0:
                                            price_liquidation_factor = 1
                                            min_price_liquidation_factor = 1
                                        else:
                                            next_missing_price_liquidation_factor = missing_price_liquidation_factor * pow(
                                                0.5, 1 / (price_recovery_time * 24 * 60))
                                            price_liquidation_factor_recovery = missing_price_liquidation_factor - next_missing_price_liquidation_factor
                                            if price_liquidation_factor_recovery < 0:
                                                print("price_liquidation_factor_recovery",
                                                      price_liquidation_factor_recovery)
                                                exit()

                                            price_liquidation_factor += price_liquidation_factor_recovery
                                            min_price_liquidation_factor = min(price_liquidation_factor,
                                                                               price_liquidation_factor)

                                            if price_liquidation_factor > 1:
                                                print("Error")
                                                exit()

                                        price = row["adjust_price"] * price_liquidation_factor

                                        if liquidation_volume != 0:
                                            liq = {"time": time, "liquidation_volume": liquidation_volume,
                                                   "price": price,
                                                   "closed": 0,
                                                   "trades": []}
                                            open_liquidations.append(liq)

                                        open_liquidations_volume = sum(
                                            [open_liquidation["liquidation_volume"] - open_liquidation["closed"] for
                                             open_liquidation in
                                             open_liquidations])

                                        drop, volume = check_max_liquidation_drop(open_liquidations, price)
                                        if max_drop < drop:
                                            max_drop = drop
                                            price_at_max_drop = price
                                            max_drop_open_volume = open_liquidations_volume

                                        market_volume = cycle_trade_volume - sum(historical_cycle_trade_volume)
                                        close_liquidation_volume = 0
                                        trade_volume = 0
                                        using_stability_pool_volume = 0
                                        stability_pool_simple_recovery = 0
                                        using_market_volume = 0
                                        stability_pool_available_volume = 0

                                        if stability_pool_initial_balance != 0:
                                            stability_pool_simple_recovery = stability_pool_simple_instance.do_tick(
                                                time,
                                                market_volume)
                                            stability_pool_available_volume = stability_pool_simple_instance.do_check_liquidation_size()
                                            using_stability_pool_volume = min(open_liquidations_volume,
                                                                              stability_pool_available_volume)
                                            close_liquidation_volume += using_stability_pool_volume
                                            trade_volume += stability_pool_simple_recovery

                                        if close_liquidation_volume < open_liquidations_volume:
                                            using_market_volume = min(
                                                open_liquidations_volume - close_liquidation_volume,
                                                market_volume - stability_pool_simple_recovery)
                                            close_liquidation_volume += using_market_volume + 1  # for Rounding issues
                                            trade_volume += using_market_volume

                                        ts_report.append({
                                            "ts": time,
                                            "price": price,
                                            "liquidation_volume": liquidation_volume,
                                            "open_liquidations": open_liquidations_volume,
                                            "market_volume": market_volume,
                                            "stability_pool_simple_recovery": stability_pool_simple_recovery,
                                            "stability_pool_available_volume": stability_pool_available_volume,
                                            "using_market_volume": using_market_volume,
                                            "close_liquidation_volume": close_liquidation_volume,
                                            "trade_volume": trade_volume,
                                            "max_drop": max_drop
                                        })

                                        if close_liquidation_volume > 0:
                                            to_delete = []
                                            for open_liquidation in open_liquidations:
                                                closed_volume = open_liquidation["closed"]
                                                open_volume = open_liquidation["liquidation_volume"] - closed_volume
                                                if open_volume > close_liquidation_volume:
                                                    open_liquidation["trades"].append(
                                                        {"time": time, "volume": close_liquidation_volume,
                                                         "price": price})
                                                    open_liquidation["closed"] += close_liquidation_volume
                                                    break
                                                else:
                                                    current_liquidation_volume = open_volume
                                                    open_liquidation["trades"].append(
                                                        {"time": time, "volume": current_liquidation_volume,
                                                         "price": price})
                                                    open_liquidation["closed"] += current_liquidation_volume
                                                    close_liquidation_volume -= current_liquidation_volume
                                                    to_delete.append(open_liquidation)

                                            if using_stability_pool_volume > 0:
                                                stability_pool_simple_instance.do_set_liquidation_size(
                                                    using_stability_pool_volume)

                                            for o in to_delete:
                                                open_liquidations.remove(o)
                                                closed_liquidations.append(copy.deepcopy(o))

                                        historical_cycle_trade_volume.append(trade_volume)
                                        xx = int(len(historical_cycle_trade_volume) - trade_every / 60)
                                        if xx > 0:
                                            historical_cycle_trade_volume = historical_cycle_trade_volume[xx:]

                                        # multiply the price by the trade slippage
                                        multiply_price_liquidation_factor = get_slippage_for_volume(
                                            cycle_trade_volume, 1 - l_incentive, trade_volume)

                                        min_multiply_price_liquidation_factor = min(
                                            min_multiply_price_liquidation_factor,
                                            multiply_price_liquidation_factor)

                                        price_liquidation_factor *= multiply_price_liquidation_factor

                                    open_volume = sum(
                                        [open_liquidation["liquidation_volume"] - open_liquidation["closed"] for
                                         open_liquidation in
                                         open_liquidations])

                                    print(" " * 10,
                                          "max_liq_volume", max_liquidation_volume,
                                          "all_liq_volume", int(all_liquidations_volume),
                                          "min_price_liq_fact", min_price_liquidation_factor,
                                          "max_drop", round(max_drop, 2),
                                          "price_at_max_drop", price_at_max_drop,
                                          "max_drop_open_volume", int(max_drop_open_volume),
                                          "min_multiply_price_liq_fact", min_multiply_price_liquidation_factor,
                                          "open_volume", int(open_volume))

                                    print()
                                    df = pd.DataFrame(ts_report)
                                    df.to_csv("results\\" + simulation_name + '.csv')

                                    report.append(
                                        {"simulation_name": simulation_name,
                                         "file_name": file_description,
                                         "file_total_volume": file_total_volume,
                                         "trade_every": trade_every,
                                         "series_std_ratio": series_std_ratio,
                                         "liquidation_incentive": l_incentive,
                                         "price_recovery_time": price_recovery_time,
                                         "volume_for_slippage_10_percents": volume_for_slippage_10_percents,
                                         "cycle_trade_volume": cycle_trade_volume,
                                         "collateral": collateral * ETH_PRICE,
                                         "recovery_halflife_retail": recovery_halflife_retail,
                                         "share_institutional": share_institutional,
                                         "stability_pool_initial_balance_ratio": s_balance,
                                         "stability_pool_initial_balance": stability_pool_initial_balance,
                                         "collateral_liquidation_factor": l_factor,
                                         "simulation volume": all_liquidations_volume,
                                         "min_multiply_price_liquidation_factor": min_multiply_price_liquidation_factor,
                                         "max_liquidation_volume": max_liquidation_volume,
                                         "min_price_liquidation_factor": min_price_liquidation_factor,
                                         "max_simulation_daily_volume": max_daily_volume,
                                         "max_drop": max_drop,
                                         "price_at_max_drop": price_at_max_drop,
                                         "max_drop_volume": max_drop_open_volume,
                                         "open_volume": open_volume})

                            df = pd.DataFrame(report)
                            df.to_csv(
                                "results\\" + file_name.replace("\\", "_") + "_" + name + "_" + "stability_report.csv")

                        df = pd.DataFrame(report)
                        df.to_csv(
                            "results\\" + file_name.replace("\\", "_") + "_" + name + "_" + "stability_report.csv")


liquidation_side = "binance-futures_ethusdt_liquidation_long"
liquidation_factor = 1  # ratio of liquidation
ETH_PRICE = 2000

all_config = {
    "dpx": {
        "series_std_ratio": 2.64,
        "volume_for_slippage_10_percentss": [350, 700],
        "trade_every": 1800,
        "collaterals": [2_000_000 / ETH_PRICE, 4_000_000 / ETH_PRICE, 8_000_000 / ETH_PRICE, 16_000_000 / ETH_PRICE],
        "liquidation_incentives": [0.1],
        "stability_pool_initial_balances": [0, 0.1, 0.25, 0.5],
        "share_institutionals": [0, 0.25, 0.5],
        "recovery_halflife_retails": [1, 5, 10],
        "price_recovery_times": [0.5, 1, 2, 4, 10],
        "l_factors": [0.5, 1, 2, 4, 6]
    },
    "gmx": {
        "series_std_ratio": 2.84,
        'volume_for_slippage_10_percentss': [150, 300],
        'trade_every': 1800,
        "collaterals": [2_000_000 / ETH_PRICE, 4_000_000 / ETH_PRICE, 8_000_000 / ETH_PRICE, 16_000_000 / ETH_PRICE],
        'liquidation_incentives': [0.1],
        "stability_pool_initial_balances": [0, 0.1, 0.25, 0.5],
        'share_institutionals': [0, 0.25, 0.5],
        'recovery_halflife_retails': [1, 5, 10],
        "price_recovery_times": [0.5, 1, 2, 4, 10],
        'l_factors': [0.5, 1, 2, 4, 6]
    },
    "gohm": {
        'series_std_ratio': 1.15,
        'volume_for_slippage_10_percentss': [20, 40],
        'trade_every': 1800,
        "collaterals": [2_000_000 / ETH_PRICE, 4_000_000 / ETH_PRICE, 8_000_000 / ETH_PRICE, 16_000_000 / ETH_PRICE],
        'liquidation_incentives': [0.1],
        'stability_pool_initial_balances': [0, 0.1, 0.25, 0.5],
        'share_institutionals': [0, 0.25, 0.5],
        'recovery_halflife_retails': [1, 5, 10],
        'price_recovery_times': [0.000001],
        'l_factors': [0.5, 1, 2, 4, 6]},
    "btc": {
        'series_std_ratio': 0.78,
        'volume_for_slippage_10_percentss': [250, 500],
        'trade_every': 1800,
        'collaterals': [5_000_000 / ETH_PRICE, 10_000_000 / ETH_PRICE, 15_000_000 / ETH_PRICE, 30_000_000 / ETH_PRICE],
        'liquidation_incentives': [0.1],
        'stability_pool_initial_balances': [0, 0.1, 0.25, 0.5],
        'share_institutionals': [0, 0.25, 0.5],
        'recovery_halflife_retails': [1, 5, 10],
        'price_recovery_times': [0.000001],
        'l_factors': [0.5, 1, 2, 4, 6]
    },
    "eth": {
        'series_std_ratio': 1,
        'volume_for_slippage_10_percentss': [400, 800],
        'trade_every': 1800,
        'collaterals': [5_000_000 / ETH_PRICE, 10_000_000 / ETH_PRICE, 15_000_000 / ETH_PRICE, 30_000_000 / ETH_PRICE],
        'liquidation_incentives': [0.1],
        'stability_pool_initial_balances': [0, 0.1, 0.25, 0.5],
        'share_institutionals': [0, 0.25, 0.5],
        'recovery_halflife_retails': [1, 5, 10],
        'price_recovery_times': [0.000001],
        'l_factors': [0.5, 1, 2, 4, 6]
    },
    "tri": {
        'series_std_ratio': 0.66,
        'volume_for_slippage_10_percentss': [750, 1500],
        'trade_every': 1800,
        'collaterals': [5_000_000 / ETH_PRICE, 10_000_000 / ETH_PRICE, 15_000_000 / ETH_PRICE, 30_000_000 / ETH_PRICE],
        'liquidation_incentives': [0.1],
        'stability_pool_initial_balances': [0, 0.1, 0.25, 0.5],
        'share_institutionals': [0, 0.25, 0.5],
        'recovery_halflife_retails': [1, 5, 10],
        'price_recovery_times': [0.000001],
        'l_factors': [0.5, 1, 2, 4, 6]
    }
}

# calc_series_std_ratio("data\\spell_eth_6_months.csv", "data\\dai_eth_6_months.csv", False)
# calc_series_std_ratio("data\\ohm_eth_3_months.csv", "data\\dai_eth_3_months.csv", True)
# calc_series_std_ratio("data\\wbtc_eth_6_months.csv", "data\\dai_eth_6_months.csv", True)
# calc_series_std_ratio("data\\arbitrum_dpx_eth_3_months.csv", "data\\arbitrum_dai_eth_3_months.csv", True)
# calc_series_std_ratio("data\\arbitrum_gmx_eth_3_months.csv", "data\\arbitrum_dai_eth_3_months.csv", True)

if __name__ == '__main__':
    files = glob.glob("data\\*eth*")
    test = Parallel(n_jobs=6)(delayed(run_simulation)(files[0], i, all_config[i]) for i in all_config)

for x in all_config:
    plot_for_html(x)