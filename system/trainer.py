from copy import deepcopy
from collections import defaultdict
import heapq
import multiprocessing as mp
import os

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as md
import datetime as dt
import numpy as np

import rules
import trader
import experts
import data
import config



class Trainer:
    def __init__(self):
        self.loaded_history = {}

    def construct_trader(self, cfg: config.Config):

        self.cfg = cfg

        if cfg.load_expert:
            pair_expert = config.deserialize_expert_from_json(cfg.load_expert)
        elif not cfg.reestimate:
            pair_expert = config.deserialize_expert_from_json('estimated_expert.json')
        elif cfg.reestimate:
            searchspace = config.constract_searchspace()
            pair_expert = self.constract_expert_from_searchspace(searchspace, cfg)

            pair_expert.show()
            self.trim_bad_experts(pair_expert, nbest=10**5, verbose=True, ndays=100)
            config.serialize_expert_to_json(filename='estimated_expert.json', expert=pair_expert)

        # p_expert.show()
        self.choose_branches(pair_expert, timeframes=cfg.timeframes, rules=cfg.rules)
        # pair_expert.show()
        self.trim_bad_experts(pair_expert, min_trades=1, nbest=cfg.nleaves, ndays=None)
        assert pair_expert.count_total_inner_experts() != 0, "empty expert"
        pair_expert.show()

        if cfg.load_weights:
            pair_expert.load_weights()
        else:
            pair_expert.set_weights(recursive=True)
            self.fit_weights(pair_expert)
            pair_expert.save_weights()

        pair_trader = trader.PairTrader(cfg)
        pair_trader.set_expert(pair_expert)

        config.serialize_expert_to_json(filename='expert.json', expert=pair_expert)

        return pair_trader

    def constract_expert_from_searchspace(self, searchspace: dict, cfg: config.Config):
        timeframe_lst = []
        for timeframe in cfg.timeframes:
            rule_cls_lst = []
            print(f'load timeframe [{timeframe}]')

            for rule in cfg.rules:
                new = [expert for expert in config.get_experts_from_searchspace(timeframe, rule)]
                print(' ' * 5 + f'{rule:<60} {len(new):>5} candidates')
                rule_cls_expert = experts.RuleClassExpert(rule)
                rule_cls_expert.set_experts(new)
                rule_cls_lst.append(rule_cls_expert)

            timeframe_expert = experts.TimeFrameExpert(timeframe)
            timeframe_expert.set_experts(rule_cls_lst)
            timeframe_lst.append(timeframe_expert)

        base, quote = cfg.pair.split('/')
        pair_expert = experts.PairExpert(base, quote)
        pair_expert.set_experts(timeframe_lst)
        return pair_expert

    def choose_branches(self, expert: experts.BaseExpert, *,
                              timeframes: list[str] = None,
                              rules: list[str] = None):

        if isinstance(expert, experts.PairExpert) and timeframes is not None:
            expert._inner_experts = [exp for exp in expert._inner_experts if exp.timeframe in timeframes]
        if isinstance(expert, experts.TimeFrameExpert) and rules is not None:
            expert._inner_experts = [exp for exp in expert._inner_experts if exp.rule in rules]
        if hasattr(expert, '_inner_experts'):
            for exp in expert._inner_experts:
                self.choose_branches(exp, timeframes=timeframes, rules=rules)

    def trim_bad_experts(self, expert: experts.BaseExpert, *, ret_dict: dict = None, verbose: bool = False, indentation: int = 0, **kwargs):
        kwargs |= {attr: getattr(expert, attr) for attr in ('base', 'quote', 'pair', 'timeframe') if hasattr(expert, attr)}
        if isinstance(expert, experts.RuleClassExpert):
            expert._inner_experts = self.best_rule_experts(expert._inner_experts, rule=expert.rule, **kwargs)
        else:
            results = mp.Manager().dict()
            kwargs |= {'verbose': verbose, 'indentation': indentation + 10}
            jobs = [mp.Process(target=self.trim_bad_experts,
                               args=(expert,),
                               kwargs=kwargs | {'ret_dict': results}) for expert in expert._inner_experts]
            for job in jobs:
                job.start()
            for job in jobs:
                job.join()
            expert._inner_experts = results.values()
        if ret_dict is not None:
            ret_dict[os.getpid()] = expert
        if verbose:
            print(f'{" " * indentation}trimmed: {expert.name}')

    def best_rule_experts(self, candidates: list[experts.RuleExpert],
                                min_trades: int = None, *,
                                trashold: float = None,
                                nbest: int = None,
                                percent: 'float (0, 1)' = None,
                                **kwargs) -> list[experts.RuleExpert]:
        nbest = nbest if percent is None else int(percent * len(candidates))
        for expert in candidates:
            self.estimate_expert(expert, **kwargs)
        if min_trades is not None:
            candidates = [expert for expert in candidates if expert.estimation['ntrades'] >= min_trades]
        candidates.sort(reverse=True, key=lambda x: x.estimation['fitness'])
        if trashold is not None:
            return [expert for expert in candidates if expert.estimation['profit'] > trashold]
        elif nbest is not None:
            return candidates[:nbest]

    def estimate_expert(self, expert: experts.RuleExpert,
                              pair: str,
                              timeframe: str,
                              ndays: int,
                              **kwargs):
        if expert.estimation['profit'] is None:
            pair_trader = trader.PairTrader(self.cfg)
            pair_expert = self.cast_to_pair_expert(expert, timeframe=timeframe, **kwargs)
            pair_expert.set_weights(recursive=True)
            pair_trader.set_expert(pair_expert)
            fitness, profit, ntrades = self.simulate_pair_trader(pair_trader, ndays=ndays)
            expert.estimation = {'profit': profit, 'ntrades': ntrades, 'fitness': fitness}

    def cast_to_pair_expert(self, expert: experts.BaseExpert,
                                  quote: str,
                                  base: str,
                                  timeframe: str = None,
                                  rule: str = None) -> experts.PairExpert:
        for rule_cls, upcast_cls, args in zip([experts.RuleExpert, experts.RuleClassExpert, experts.TimeFrameExpert],
                                              [experts.RuleClassExpert, experts.TimeFrameExpert, experts.PairExpert],
                                              [(rule,), (timeframe,), (quote, base)]):
            if isinstance(expert, rule_cls):
                assert None not in args, f'No arguments for {upcast_cls}'
                temp = upcast_cls(*args)
                temp.set_experts([expert])
                expert = temp
                expert.set_weights()
        return expert

    def simulate_pair_trader(self, pair_trader: trader.PairTrader, ndays: int, *, display: bool = False):
        def load_history(pair: str, timeframe: str) -> pd.DataFrame:
            if (pair, timeframe) not in self.loaded_history:
                filename = f"data/test_data/{pair.replace('/', '')}/{timeframe}.csv"
                self.loaded_history[(pair, timeframe)] = pd.read_csv(filename)
            return self.loaded_history[(pair, timeframe)]

        def construct_data(pair_trader: trader.PairTrader, ndays: int):
            init_data = data.DataMaintainer()
            new_data = {}
            start_time = load_history(pair_trader.pair, '1d')['Open time'].iloc[-ndays]
            for timeframe in pair_trader.timeframes:
                df = load_history(pair_trader.pair, timeframe)
                split = df['Open time'].searchsorted(start_time) - 1
                init, new = df.iloc[max(split-1000, 0): split].values.T, df.iloc[split:].values
                mapping = {key: val for key, val in zip(df, init)}
                init_data.add(mapping, location=[timeframe, 'Init'])
                new_data[timeframe] = new
            return init_data, new_data

        init_data, new_data = construct_data(pair_trader, ndays)
        pair_trader.set_data(init_data)

        updates = defaultdict(dict) # close time -> update
        for timeframe, rows in new_data.items():
            for row in rows:
                close_time = row[6]
                updates[close_time][timeframe] = row

        cut = 999999999 if display else 24*ndays
        for close_time, update in list(sorted(updates.items()))[:cut]:
            pair_trader.update(update)
            pair_trader.act()

        if display:
            self.show_trades(pair_trader, new_data)
        return pair_trader.fitness(), pair_trader.evaluate_profit(), len(pair_trader.trades)

    def fit_weights(self, expert: experts.BaseExpert, pair='BTC/USDT', population=3, nchildren=2, indentation=0, **kwargs):
        def estimate_trader(pair_trader: trader.PairTrader, *, ret_dict = None) -> float:
            fitness, profit, ntrades = self.simulate_pair_trader(pair_trader, ndays=350)
            ret_dict[hash(pair_trader)] = (fitness, profit)

        def change_weights(weights: np.array):
            sigma = lr * np.exp(-decay * epoch)
            return weights + np.random.normal(size=weights.shape, scale=sigma)

        def parallel_estimation(traders: list[trader.PairTrader]):
            results = mp.Manager().dict()
            jobs = [mp.Process(target=estimate_trader, args=(trader,), kwargs={'ret_dict': results}) for trader in traders]
            for job in jobs:
                job.start()
            for job in jobs:
                job.join()
            return [results[hash(trader)] for trader in traders]

        kwargs |= {attr: getattr(expert, attr) for attr in ('quote', 'base', 'timeframe', 'rule') if hasattr(expert, attr)}
        if not isinstance(expert, experts.RuleClassExpert):
            for exp in expert._inner_experts:
                self.fit_weights(exp, indentation=indentation+10, **kwargs)

        if len(expert._inner_experts) > 1:
            print(' ' * indentation + f'{expert.name}')
            lr, decay = 1, .3
            min_trades = 10
            parents = [expert.get_weights()]

            for epoch in range(self.cfg.nepochs):
                children = []
                for weights in parents:
                    children += [change_weights(weights) for _ in range(nchildren)]
                parents += children

                traders = []
                for weights in parents:
                    exp = deepcopy(expert)
                    exp.set_weights(weights)
                    exp = self.cast_to_pair_expert(exp, **kwargs)
                    tr = trader.PairTrader(self.cfg)
                    tr.set_expert(exp)
                    traders.append(tr)

                estimations = parallel_estimation(traders)

                results = zip(estimations, parents)
                results = heapq.nlargest(population, results, key=lambda x: x[0])
                parents = [weights for est, weights in results]

                (best_fitness, best_profit), best_weights = max(results, key=lambda x: x[0])
                if expert.estimation['fitness'] is None or expert.estimation['fitness'] < best_fitness:
                    expert.estimation['fitness'] = best_fitness
                    expert.set_weights(best_weights)

                print(' ' * (indentation + 100) + f'[ {epoch+1:>3} / {self.cfg.nepochs:<3} ] fitness: {best_fitness:.2f} profit: {best_profit:.2f} %')

    def show_trades(self, pair_trader: trader.PairTrader, new_data: dict):
        def config_axs(*axs):
            for ax in axs:
                ax.grid(True)
                ax.margins(x=.0)
                xfmt = md.DateFormatter('%Y-%b')
                ax.xaxis.set_major_formatter(xfmt)

        pair_trader.show_evaluation()

        timeframe = pair_trader.min_timeframe
        close = new_data[timeframe][:, 4] # Close price
        time = new_data[timeframe][:, 6] # Close time
        buy_time, buy_price = pair_trader.times[::2], [trade[2] for trade in pair_trader.trades[::2]]
        sell_time, sell_price = pair_trader.times[1::2], [trade[2] for trade in pair_trader.trades[1::2]]

        fig, axs = plt.subplots(nrows=3, figsize=(19.2, 10.8), dpi=100)
        fig.tight_layout()
        ax1, ax2, ax3 = axs.reshape(-1)
        config_axs(ax1, ax2, ax3)

        _time = lambda x: [dt.datetime.fromtimestamp(ts//1000) for ts in x]
        ax1.plot(_time(time), close, color='black', linewidth=1)
        ax1.scatter(_time(buy_time), buy_price, color='blue')
        ax1.scatter(_time(sell_time), sell_price, color='red')
        ax1.set_title("Price")

        ax2.plot(_time(pair_trader.times), pair_trader._profits, linestyle=':')
        ax2.set_title("Profit")

        estimations = pair_trader.estimations
        ax3.plot(_time(time[:len(estimations)]), estimations)
        ax3.set_title("Signal")

        plt.show()