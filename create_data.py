import pandas as pd
from stock_env.apps import config
from preprocessor.yahoodownloader import YahooDownloader
from preprocessor.preprocessors import FeatureEngineer, data_split
import itertools
import argparse
from preprocessor.process_traj import trajectory
import numpy as np
import pickle
import os

def create_data(variant):
    #Create datasets
    if variant['dataset']=="dow":
        df = YahooDownloader(start_date = '2009-01-01',
                              end_date = '2020-09-24',
                             ticker_list = config.DOW_TICKER).fetch_data()
    elif variant['dataset']=="hightech":
        df = YahooDownloader(start_date = '2006-10-20',
                             end_date = '2013-11-21',
                             ticker_list = config.HighTech_TICKER).fetch_data()
    elif variant['dataset'] == "ndx":
        df = YahooDownloader(start_date = '2009-01-01',
                            end_date = '2021-12-31',
                            ticker_list = config.NDX_TICKER).fetch_data()
    elif variant['dataset'] == "mdax":
        df = YahooDownloader(start_date = '2009-01-01',
                            end_date = '2021-12-31',
                            ticker_list = config.MDAX_TICKER).fetch_data()
    elif variant['dataset'] == "csi":
        df = YahooDownloader(start_date = '2009-01-01',
                            end_date = '2021-12-31',
                            ticker_list = config.CSI_TICKER).fetch_data()

    # Add technical indicator
    fe = FeatureEngineer(
                        use_technical_indicator=True,
                        tech_indicator_list = config.TECHNICAL_INDICATORS_LIST,
                        use_turbulence=True
    )

    processed = fe.preprocess_data(df)

    list_ticker = processed["tic"].unique().tolist()
    list_date = list(pd.date_range(processed['date'].min(),processed['date'].max()).astype(str))
    combination = list(itertools.product(list_date,list_ticker))

    processed_full = pd.DataFrame(combination,columns=["date","tic"]).merge(processed,on=["date","tic"],how="left")
    processed_full = processed_full[processed_full['date'].isin(processed['date'])]
    processed_full = processed_full.sort_values(['date','tic'])
    processed_full = processed_full.fillna(0)
    
    # Split train and test datasets
    if variant['dataset'] == "dow":
        train = data_split(processed_full, '2009-01-01','2019-01-01')
        trade = data_split(processed_full, '2019-01-01','2020-09-24')
    elif variant['dataset'] == "hightech":
        train = data_split(processed_full, '2006-10-20','2012-11-16')
        trade = data_split(processed_full, '2012-11-16','2013-11-21')
    else:
        train = data_split(processed_full, '2009-01-01','2020-09-01')
        trade = data_split(processed_full, '2020-09-01','2021-12-31')

    if not os.path.exists("datasets"):
        os.makedirs("datasets")

    train.to_csv("datasets/"+variant['dataset']+"_train.csv")
    trade.to_csv("datasets/"+variant['dataset']+"_trade.csv")


    ###################Create suboptimal trajectories########################

    train = pd.read_csv("datasets/"+variant['dataset']+"_train.csv", index_col=[0])

    stock_dimension = len(train.tic.unique())
    state_space = 4 * stock_dimension + len(config.TECHNICAL_INDICATORS_LIST) * stock_dimension
    print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")

    # Pass the reward mode to env_kwargs
    env_kwargs = {
        "state_space": state_space,
        "stock_dim": stock_dimension,
        "tech_indicator_list": config.TECHNICAL_INDICATORS_LIST,
        "action_space": stock_dimension,
        "reward_mode": variant['reward']  # Added this line
    }
    
    env = trajectory(df=train, dataset=variant['dataset'], **env_kwargs)

    def traj_generator(env, episode):
        ob = env.reset()
        obs = []
        rews = []
        term = []
        acs = []

        while True:
            next_state, rew, new, ac = env.step(episode)
            obs.append(ob)
            term.append(new)
            acs.append(ac)
            rews.append(rew)
            ob = next_state

            if new:
                break

        obs = np.array(obs)
        rews = np.array(rews)
        term = np.array(term)
        acs = np.array(acs)
        print(f"Episode {episode} Reward Sum: {np.sum(rews)}")
        traj = {"observations": obs, "rewards": rews, "dones": term, "actions": acs}
        return traj

    paths = []

    for i in range(5):
        traj = traj_generator(env, i)
        paths.append(traj)

    if not os.path.exists("trajectory"):
        os.makedirs("trajectory")

    # Save with reward mode in filename to avoid confusion
    name = f'{"trajectory/"+variant["dataset"]+"_"+variant["reward"]+"_traj"}'
    with open(f'{name}.pkl', 'wb') as f:
        pickle.dump(paths, f)

    print(f"Created trajectories with {variant['reward']} reward shaping at {name}.pkl")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='hightech')
    # Added argument for reward
    parser.add_argument('--reward', type=str, default='none', choices=['none', 'atr', 'sharpe'], help='Reward shaping mode: none, atr, or sharpe') 

    args = parser.parse_args()
    create_data(variant=vars(args))