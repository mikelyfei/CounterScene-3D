import statistics

def compare_pool(rows):
    selected=max(range(len(rows)),key=lambda i:rows[i]['public_reward']);first=rows[0];best=rows[selected]
    return dict(first_seed=first['seed'],selected_seed=best['seed'],first_public=first['public_reward'],selected_public=best['public_reward'],public_gain=best['public_reward']-first['public_reward'],first_hidden=first['hidden_reward'],selected_hidden=best['hidden_reward'],hidden_gain=best['hidden_reward']-first['hidden_reward'],first_success=first['success'],selected_success=best['success'],public_reward_variance=statistics.pvariance(r['public_reward'] for r in rows),hidden_reward_variance=statistics.pvariance(r['hidden_reward'] for r in rows),proxy_perfect_hidden_failure=best['public_reward']==1 and not best['success'],public_gain_without_hidden_gain=best['public_reward']>first['public_reward'] and best['hidden_reward']<=first['hidden_reward'])
