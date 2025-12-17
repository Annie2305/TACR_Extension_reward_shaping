import gym
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from gym import spaces
from gym.utils import seeding
import os

matplotlib.use("Agg")

class StockPortfolioEnv(gym.Env):
    """Stock trading environment for OpenAI gym"""
    metadata = {"render.modes": ["human"]}

    def __init__(
            self,
            df,
            stock_dim,
            initial_amount,
            transaction_cost,
            state_space,
            action_space,
            tech_indicator_list,
            dataset,
            turbulence_threshold=None,
            mode="",
            reward_mode="none", 
            lookback=252,
            day=0,
    ):
        self.dataset = dataset
        self.day = day
        self.lookback = lookback
        self.df = df
        self.stock_dim = stock_dim
        self.initial_amount = initial_amount
        self.transaction_cost = transaction_cost
        self.mode = mode
        self.reward_mode = reward_mode 
        self.state_space = state_space
        self.action_space = action_space
        self.tech_indicator_list = tech_indicator_list

        # action_space normalization and shape is self.stock_dim
        self.action_space = spaces.Box(low=0, high=1, shape=(self.action_space,))

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_space,)
        )
        # load data from a pandas dataframe
        self.data = self.df.loc[self.day, :]

        if dataset=="kdd":
            self.state = (
                    self.data.open.values.tolist()
                    + self.data.high.values.tolist()
                    + self.data.low.values.tolist()
                    + self.data.close.values.tolist()
            )
        else:
            self.state = (
                    self.data.open.values.tolist()
                    + self.data.high.values.tolist()
                    + self.data.low.values.tolist()
                    + self.data.close.values.tolist()
                    + sum([self.data[tech].values.tolist() for tech in self.tech_indicator_list],[],)
            )

        self.terminal = False
        self.turbulence_threshold = turbulence_threshold
        # initalize state
        self.portfolio_value = self.initial_amount
        self.turbulence = 0
        self.pre_weights = 0
        # memorize portfolio value each step
        self.asset_memory = [self.initial_amount]
        # memorize portfolio return each step
        self.portfolio_return_memory = [0]
        self.actions_memory = [[1 / self.stock_dim] * self.stock_dim]
        self.date_memory = [self.data.date.unique()[0]]

    def step(self, actions):
        self.terminal = self.day >= len(self.df.index.unique()) - 1

        if self.terminal:
            # --- 1. Create Subfolder ---
            folder_name = self.reward_mode.capitalize() if self.reward_mode else "None"
            save_path = os.path.join("results", folder_name)
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            # --- 2. Calculate Metrics ---
            
            # A. Sharpe
            df_daily_return = pd.DataFrame(self.portfolio_return_memory)
            df_daily_return.columns = ["daily_return"]
            df_daily_return_adj = df_daily_return.copy()
            df_daily_return_adj[1:] -= 0.02 / 365  # bank interest
            sharpe = 0
            if df_daily_return_adj["daily_return"].std() != 0:
                sharpe = (
                        (252 ** 0.5)
                        * df_daily_return_adj["daily_return"].mean()
                        / df_daily_return_adj["daily_return"].std())
            
            # B. Sortino
            sortino = 0
            downside_returns = df_daily_return_adj.loc[df_daily_return_adj['daily_return'] < 0, 'daily_return']
            if len(downside_returns) > 0 and downside_returns.std() != 0:
                downside_std = downside_returns.std()
                sortino = (
                    (252 ** 0.5) 
                    * df_daily_return_adj["daily_return"].mean() 
                    / downside_std
                )

            # C. MDD (Max Drawdown)
            df_asset = pd.DataFrame(self.asset_memory)
            df_asset.columns = ["asset"]
            roll_max = df_asset["asset"].cummax()
            drawdown = df_asset["asset"] / roll_max - 1.0
            max_drawdown = drawdown.min()

            # D. Turnover Rate
            action_df = pd.DataFrame(self.actions_memory)
            turnover_df = action_df.diff().abs().sum(axis=1) / 2
            turnover_rate = turnover_df.mean()

            # --- 3. Save Summary Statistics to CSV ---
            # FIX: Filename now includes dataset name to prevent overwriting
            summary_data = {
                "Dataset": [self.dataset],
                "Reward_Mode": [folder_name],
                "Initial_Asset": [self.asset_memory[0]],
                "Final_Asset": [int(self.portfolio_value)],
                "Sharpe_Ratio": [round(sharpe, 4)],
                "Sortino_Ratio": [round(sortino, 4)],
                "MDD": [round(max_drawdown, 4)],
                "Turnover_Rate": [round(turnover_rate, 4)]
            }
            df_summary = pd.DataFrame(summary_data)
            
            # UNIQUE FILENAME: results_summary_hightech.csv, results_summary_dow.csv, etc.
            summary_csv_path = os.path.join(save_path, f"results_summary_{self.dataset}.csv")
            
            df_summary.to_csv(summary_csv_path, index=False)
            print(f"Summary results saved to {summary_csv_path}")

            # --- 4. Print Table to Terminal ---
            print("\n" + "="*105)
            print(f"{'Dataset':<10} | {'Reward':<10} | {'Init Asset':<12} | {'Final Asset':<12} | {'Sharpe':<8} | {'Sortino':<8} | {'MDD':<8} | {'Turnover':<8}")
            print("-" * 105)
            print(f"{self.dataset:<10} | {folder_name:<10} | {self.asset_memory[0]:<12} | {int(self.portfolio_value):<12} | {sharpe:.3f}    | {sortino:.3f}    | {max_drawdown:.4f}   | {turnover_rate:.4f}")
            print("="*105 + "\n")

            # --- 5. Plotting ---
            date_objects = pd.to_datetime(self.date_memory)

            def style_plot(ax):
                ax.grid(True) 
                ax.xaxis.set_major_locator(mdates.AutoDateLocator())
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

            # PLOT 1: Cumulative Reward
            plt.figure(figsize=(12, 6))
            ax1 = plt.gca()
            ax1.plot(date_objects, df_daily_return.daily_return.cumsum(), "r")
            ax1.set_title(f"Cumulative Reward - {self.dataset} ({folder_name})", fontsize=16)
            ax1.set_xlabel("Date", fontsize=12)
            ax1.set_ylabel("Cumulative Return", fontsize=12)
            style_plot(ax1)
            plt.savefig(os.path.join(save_path, self.dataset + "_cumulative_reward.png"))
            plt.close()

            # PLOT 2: Daily Rewards
            plt.figure(figsize=(12, 6))
            ax2 = plt.gca()
            ax2.plot(date_objects, self.portfolio_return_memory, "r") 
            ax2.set_title(f"Daily Reward - {self.dataset} ({folder_name})", fontsize=16)
            ax2.set_xlabel("Date", fontsize=12)
            ax2.set_ylabel("Reward", fontsize=12)
            style_plot(ax2)
            plt.savefig(os.path.join(save_path, self.dataset + "_rewards.png"))
            plt.close()

            if self.mode == "test":
                df_actions = self.save_action_memory()
                df_actions.to_csv(os.path.join(save_path, "actions_{}.csv".format(self.mode)))

                df_asset = self.save_asset_memory()
                df_asset.to_csv(os.path.join(save_path, "{}_asset_{}.csv".format(self.dataset, self.mode)))

                # PLOT 3: Asset Value
                plt.figure(figsize=(12, 6))
                ax3 = plt.gca()
                ax3.plot(date_objects, df_asset["asset"], "r")
                
                plt.suptitle(f"Portfolio Value Over Time - {self.dataset} ({folder_name})", fontsize=16, y=0.96)
                subtitle = f"Final Asset: ${int(self.portfolio_value):,} | Sharpe: {sharpe:.3f} | Sortino: {sortino:.3f} | MDD: {max_drawdown:.2%} | Turnover: {turnover_rate:.3f}"
                ax3.set_title(subtitle, fontsize=12, pad=10, color='black')

                ax3.set_xlabel("Date", fontsize=12)
                ax3.set_ylabel("Asset Value ($)", fontsize=12)
                style_plot(ax3)
                plt.tight_layout()
                plt.savefig(os.path.join(save_path, "{}_account_value_{}.png".format(self.dataset, self.mode)))
                plt.close()

            return self.state, self.reward, self.terminal, {}

        else:
            weights = actions

            if self.turbulence_threshold is not None:
                if self.turbulence >= self.turbulence_threshold:
                    weights = np.zeros(len(weights), dtype=float)

            self.actions_memory.append(weights)
            last_day_memory = self.data

            # load next state
            self.day += 1
            self.data = self.df.loc[self.day, :]

            if self.turbulence_threshold is not None:
                if len(self.df.tic.unique()) == 1:
                    self.turbulence = self.data["turbulence"]
                elif len(self.df.tic.unique()) > 1:
                    self.turbulence = self.data["turbulence"].values[0]

            if self.dataset == "kdd":
                self.state = (
                        self.data.open.values.tolist()
                        + self.data.high.values.tolist()
                        + self.data.low.values.tolist()
                        + self.data.close.values.tolist()
                )
            else:
                self.state = (
                        self.data.open.values.tolist()
                        + self.data.high.values.tolist()
                        + self.data.low.values.tolist()
                        + self.data.close.values.tolist()
                        + sum([self.data[tech].values.tolist() for tech in self.tech_indicator_list], [], )
                )

            # calcualte portfolio return
            # Equation (19) : Portfolio value
            portfolio_return = sum(
                ((self.data.close.values / last_day_memory.close.values) - 1) * weights - self.transaction_cost * abs(weights - self.actions_memory[-2]))

            # update portfolio value
            new_portfolio_value = self.portfolio_value * (1 + portfolio_return)
            self.portfolio_value = new_portfolio_value

            # save into memory
            self.portfolio_return_memory.append(portfolio_return)
            self.date_memory.append(self.data.date.unique()[0])
            self.asset_memory.append(new_portfolio_value)

            # =======================================================
            #                 REWARD SHAPING LOGIC
            # =======================================================
            
            base_reward = sum(((self.data.close.values / last_day_memory.close.values) - 1) * weights)

            if self.reward_mode == "atr":
                high = self.data.high.values
                low = self.data.low.values
                close = self.data.close.values
                prev_close = last_day_memory.close.values

                true_range = np.maximum(
                    high - low,
                    np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)),
                )
                atr = true_range.mean()
                
                avg_close_price = close.mean()
                if avg_close_price == 0:
                    vol_t = 0
                else:
                    vol_t = atr / avg_close_price

                alpha = -1.0
                self.reward = base_reward + alpha * vol_t

            elif self.reward_mode == "sharpe":
                r_f = 0.0001
                eps = 1e-6
                
                window = 20
                if len(self.portfolio_return_memory) < 2:
                    sigma_t = 1.0 
                else:
                    recent_returns = self.portfolio_return_memory[-window:]
                    sigma_t = np.std(recent_returns) + eps
                
                self.reward = (base_reward - r_f) / sigma_t

            else:
                self.reward = base_reward

        return self.state, self.reward, self.terminal, {}

    def reset(self):
        self.asset_memory = [self.initial_amount]
        self.day = 0
        self.data = self.df.loc[self.day, :]
        self.turbulence = 0
        if self.dataset == "kdd":
            self.state = (
                    self.data.open.values.tolist()
                    + self.data.high.values.tolist()
                    + self.data.low.values.tolist()
                    + self.data.close.values.tolist()
            )
        else:
            self.state = (
                    self.data.open.values.tolist()
                    + self.data.high.values.tolist()
                    + self.data.low.values.tolist()
                    + self.data.close.values.tolist()
                    + sum([self.data[tech].values.tolist() for tech in self.tech_indicator_list], [], )
            )

        self.portfolio_value = self.initial_amount
        self.terminal = False
        self.portfolio_return_memory = [0]
        self.actions_memory = [[1 / self.stock_dim] * self.stock_dim]
        self.date_memory = [self.data.date.unique()[0]]
        return self.state

    def render(self, mode="human"):
        return self.state

    def softmax_normalization(self, actions):
        numerator = np.exp(actions)
        denominator = np.sum(np.exp(actions))
        softmax_output = numerator / denominator
        return softmax_output

    def save_action_memory(self):
        date_list = self.date_memory
        df_date = pd.DataFrame(date_list)
        df_date.columns = ["date"]

        action_list = self.actions_memory
        df_actions = pd.DataFrame(action_list)
        df_actions.columns = self.data.tic.values
        df_actions.index = df_date.date
        return df_actions

    def save_asset_memory(self):
        date_list = self.date_memory
        df_date = pd.DataFrame(date_list)
        df_date.columns = ["date"]

        asset_list = self.asset_memory
        df_asset = pd.DataFrame(asset_list)
        df_asset.columns = ["asset"]
        df_asset.index = df_date.date
        return df_asset

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]