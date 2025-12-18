import numpy as np
import pandas as pd

class trajectory:

    def __init__(
            self,
            dataset,
            df,
            stock_dim,
            state_space,
            action_space,
            tech_indicator_list,
            reward_mode="none",  # Added reward_mode parameter
            day=0,
    ):

        self.dataset = dataset
        self.day = day
        self.df = df
        self.stock_dim = stock_dim
        self.state_space = state_space
        self.action_space = action_space
        self.tech_indicator_list = tech_indicator_list
        self.reward_mode = reward_mode # Store the mode

        self.data = self.df.loc[self.day, :]
        self.state = (
                self.data.open.values.tolist()
                + self.data.high.values.tolist()
                + self.data.low.values.tolist()
                + self.data.close.values.tolist()
                + sum(
            [
                self.data[tech].values.tolist()
                for tech in self.tech_indicator_list
            ],
            [],
        )
        )
        self.terminal = False

    def step(self, i):
        # print(self.day)
        self.terminal = self.day >= len(self.df.index.unique()) - 1
        # print(actions)
        if self.terminal:
            return self.state, 0, self.terminal, np.zeros(self.action_space, dtype=float)

        else:
            last_day_memory = self.data
            # load next state
            self.day += 1
            self.data = self.df.loc[self.day, :]
            self.state = (
                    self.data.open.values.tolist()
                    + self.data.high.values.tolist()
                    + self.data.low.values.tolist()
                    + self.data.close.values.tolist()
                    + sum(
                [
                    self.data[tech].values.tolist()
                    for tech in self.tech_indicator_list
                ],
                [],
            )
            )

            # --- Greedy Strategy for Action Generation ---
            portion = (self.data.close.values / last_day_memory.close.values)
            bc = []
            for j in portion:
                bc.append(np.exp(j * (i + 1)))

            weights = self.softmax_normalization(bc)
            weights[np.isnan(weights)] = 1.

            # Basic Portfolio Return
            portfolio_return = sum(
                ((self.data.close.values / last_day_memory.close.values) - 1) * weights
            )

            # =======================================================
            #                 REWARD SHAPING
            # =======================================================
            
            # --------- 1. ATR-based shaping ---------
            if self.reward_mode == "atr":
                high = self.data.high.values
                low = self.data.low.values
                prev_close = last_day_memory.close.values

                # Calculate True Range
                true_range = np.maximum(
                    high - low,
                    np.maximum(abs(high - prev_close), abs(low - prev_close)),
                )
                atr = true_range.mean()

                close_price = self.data.close.values.mean()
                
                # Protect against division by zero
                if close_price == 0:
                    Vol_t = 0
                else:
                    Vol_t = atr / close_price

                alpha = -1.0 # Penalize volatility
                self.reward = portfolio_return + alpha * Vol_t

            # --------- 2. Sharpe-like shaping ---------
            elif self.reward_mode == "sharpe":
<<<<<<< HEAD
                r_f = 0.02/365
=======
                r_f = 0.02 / 252  # Daily risk-free rate
>>>>>>> ef86315 (update the sharpe-based reward mode results plot and table.)
                eps = 1e-6
                
                # Calculate historical market volatility up to this point
                # We look back 20 days to calculate the rolling std deviation
                start_idx = max(0, self.day - 20)
                end_idx = self.day
                
                if end_idx - start_idx < 2:
                    # Not enough data for std dev
                    sigma_t = 1.0 
                else:
                    # Get data slice for calculation
                    # Since self.df index is factorized (0,1,2...), we can slice loc directly
                    # We need to compute returns for the window
                    period_data = self.df.loc[start_idx:end_idx]
                    
                    # Pivot to get (Date x Ticker) close prices
                    closes = period_data.pivot_table(index='date', columns='tic', values='close')
                    
                    # Calculate daily returns
                    returns = closes.pct_change().dropna()
                    
                    # Average return across all stocks (Market Proxy)
                    market_returns = returns.mean(axis=1)
                    
                    sigma_t = np.std(market_returns) + eps

                self.reward = (portfolio_return - r_f) / sigma_t

            # --------- 3. No shaping (Default) ---------
            else:
                self.reward = portfolio_return

        return self.state, self.reward, self.terminal, weights

    def reset(self):
        self.day = 0
        self.data = self.df.loc[self.day, :]
        self.state = (
                self.data.open.values.tolist()
                + self.data.high.values.tolist()
                + self.data.low.values.tolist()
                + self.data.close.values.tolist()
                + sum(
            [
                self.data[tech].values.tolist()
                for tech in self.tech_indicator_list
            ],
            [],
        )

        )
        self.terminal = False
        return self.state

    def softmax_normalization(self, actions):
        actions = np.clip(actions, 0, 709)
        numerator = np.exp(actions)
        denominator = np.sum(np.exp(actions))
        softmax_output = numerator / denominator
        return softmax_output
