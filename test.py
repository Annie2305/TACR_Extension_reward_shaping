import torch
import argparse
import pandas as pd
import random
import numpy as np
import pickle
from stock_env.apps import config
from stock_env.allocation.env_portfolio import StockPortfolioEnv
from tac.evaluation.evaluate_episodes import eval_test
from tac.models.transformer_actor import TransformerActor
import torch.backends.cudnn as cudnn
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def experiment(variant):
    device = variant.get('device', 'cuda')

    env_name, dataset = variant['env'], variant['dataset']
    reward_mode = variant['reward']  # Get reward mode from args

    # FIX 1: Update group_name to match train.py so we load the correct model file
    group_name = f'{env_name}-{dataset}-{reward_mode}'

    train = pd.read_csv("datasets/" + dataset+"_train.csv", index_col=[0])
    trade = pd.read_csv("datasets/" + dataset + "_trade.csv", index_col=[0])
    max_ep_len = train.index[-1]

    # FIX 2: Load the specific reward-shaped trajectory file used during training
    dataset_path = f'{"trajectory/" + dataset + "_" + reward_mode + "_traj.pkl"}'
    
    print(f"Loading trajectories from: {dataset_path}")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Trajectory file {dataset_path} not found. Run create_data.py with --reward {reward_mode}")

    with open(dataset_path, 'rb') as f:
        trajectories = pickle.load(f)
        
    state_space = trajectories[0]['observations'].shape[1]
    stock_dimension = len(train.tic.unique())

    print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")

    turbulence_threshold = 100 if dataset == "dow" else None
    
    env_kwargs = {
        "dataset": dataset,
        "initial_amount": 1000000,
        "transaction_cost": 0.0025,
        "state_space": state_space,
        "stock_dim": stock_dimension,
        "tech_indicator_list": config.TECHNICAL_INDICATORS_LIST,
        "action_space": stock_dimension,
        "mode": "test",
        "turbulence_threshold": turbulence_threshold,
        "reward_mode": reward_mode # Pass reward mode for consistency
    }

    env = StockPortfolioEnv(df=trade, **env_kwargs)

    seed = variant['seed']
    env.seed(seed)
    env.action_space.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(seed)

    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    states = []
    for path in trajectories:
        states.append(path['observations'])

    states = np.concatenate(states, axis=0)
    state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6

    u = variant['u']

    model = TransformerActor(
        state_dim=state_dim,
        act_dim=act_dim,
        max_length=u,
        max_ep_len=max_ep_len,
        hidden_size=variant['embed_dim'],
        n_layer=variant['n_layer'],
        n_head=variant['n_head'],
        n_inner=4 * variant['embed_dim'],
        activation_function=variant['activation_function'],
        n_positions=1024,
        resid_pdrop=variant['dropout'],
        attn_pdrop=variant['dropout'])

    # FIX 3: Load the specific model file
    model_path = group_name + '.pt'
    print(f"Loading model weights from: {model_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} not found. Did you run train.py with --reward {reward_mode}?")
        
    model.load_state_dict(torch.load(model_path))

    eval_test(
        env,
        state_dim,
        act_dim,
        model,
        max_ep_len=max_ep_len,
        state_mean=state_mean,
        state_std=state_std,
        device=device
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='csi')
    # FIX 4: Add reward argument
    parser.add_argument('--reward', type=str, default='none', choices=['none', 'atr', 'sharpe'], help='Reward shaping mode')
    
    parser.add_argument('--env', type=str, default='stock')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--u', type=int, default=20)
    parser.add_argument('--pct_traj', type=float, default=1.)
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--n_layer', type=int, default=5)
    parser.add_argument('--n_head', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--activation_function', type=str, default='relu')
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()
    experiment(variant=vars(args))