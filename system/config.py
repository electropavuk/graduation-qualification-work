"""Module for config construction and parsing."""

import json
import collections
import argparse
from itertools import product

import numpy as np

import experts
import indicators
import rules



class Config:
    def __init__(self):
        pair: str = 'BTC/USDT'
        timeframe: str = ''
        rule: str = 'MovingAverageCrossoverRule'
        nleaves: int = 3
        reestimate: bool = False
        load_expert: str = ''
        load_weights: str = bool
        nepochs: int = 3
        all_rules: bool = False
        threshold: bool = False
        fee: float = 0.0075

    def to_json(self) -> str:
        return json.dumps({
            "pair": self.pair,
            "timeframes": self.timeframes,
            "rules": self.rules,
            "nleaves": self.nleaves,
            "reestimate": self.reestimate,
            "load_expert": self.load_expert,
            "load_weights": self.load_weights,
            "nepochs": self.nepochs,
            "threshold": self.threshold,
            "fee": self.fee,
        }, indent=4)

    @classmethod
    def new_arg_parser(cls) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        parser.add_argument(
            "--pair", type=str,
            default='BTC/USDT',
            help="Currency pair.",
        )
        parser.add_argument(
            "--timeframe", type=str,
            default='',
            help="Trading timeframe.",
        )
        parser.add_argument(
            "--rule", type=str,
            default='',
            help="Trading rule.",
        )
        parser.add_argument(
            "--nleaves", type=int,
            default=3,
            help="Maximum number of RuleExperts.",
        )
        parser.add_argument(
            "--reestimate", 
            action='store_true',
            help="Optimize expert from scratch.",
        )
        parser.add_argument(
            "--load_expert", type=str,
            default='',
            help="Filename. Load expert from json file.",
        )
        parser.add_argument(
            "--load_weights", 
            action='store_true',
            help="Load expert weights from file.",
        )
        parser.add_argument(
            "--nepochs", type=int,
            default=3,
            help="Fit weights for n epochs.",
        )
        parser.add_argument(
            "--all_rules",
            action='store_true',
            help="Use all available rules. (\
        'MovingAverageCrossoverRule',\
        'ExponentialMovingAverageCrossoverRule',\
        'RelativeStrengthIndexTrasholdRule',\
        'TripleExponentialDirectionChangeRule',\
        'IchimokuKinkoHyoTenkanKijunCrossoverRule',\
        'IchimokuKinkoHyoSenkouASenkouBCrossoverRule',\
        'IchimokuKinkoHyoChikouCrossoverRule',\
        'BollingerBandsLowerUpperCrossoverRule',\
        'BollingerBandsLowerMidCrossoverRule',\
        'BollingerBandsUpperMidCrossoverRule',\
        'MovingAverageConvergenceDivergenceSignalLineCrossoverRule')",
        )
        parser.add_argument(
            "--best_rules", 
            action='store_true',
            help="Use the rules: ('MovingAverageCrossoverRule',\
                'RelativeStrengthIndexTrasholdRule',\
                'TripleExponentialDirectionChangeRule',\
                'BollingerBandsLowerMidCrossoverRule',\
                'BollingerBandsUpperMidCrossoverRule',\
                'MovingAverageConvergenceDivergenceSignalLineCrossoverRule')",
        )
        parser.add_argument(
            "--best_timeframes",
            action='store_true',
            help="Use Timeframes: (1h, 4h, 1d)."
        )
        parser.add_argument(
            "--threshold", type=float,
            default=.2,
            help="System sensativity. If confidence is above threshold the trade is made. Should be at range (0, 1)",
        )
        parser.add_argument(
            "--fee", type=float,
            default=0.0075,
            help="Trading fee. Should be at range (0, 1)",
        )
        return parser

    @classmethod
    def from_args(cls):
        """Creates Config from command line arguments."""

        c = cls()

        parser = cls.new_arg_parser()
        args = parser.parse_args()

        c.pair = args.pair
        c.nleaves = args.nleaves
        c.reestimate = args.reestimate
        c.load_expert = args.load_expert
        c.load_weights = args.load_weights
        c.nepochs = args.nepochs
        c.all_rules = args.all_rules
        c.threshold = args.threshold
        c.fee = args.fee
        
        c.timeframes = ['4h']

        if args.rule:
            c.rules = [args.rule]
        elif args.best_rules:
            c.rules = [
                'MovingAverageCrossoverRule',
                'RelativeStrengthIndexTrasholdRule',
                'TripleExponentialDirectionChangeRule',
                'BollingerBandsLowerMidCrossoverRule',
                'BollingerBandsUpperMidCrossoverRule',
                'MovingAverageConvergenceDivergenceSignalLineCrossoverRule',
            ]
        elif args.all_rules:
            c.rules = [
                'MovingAverageCrossoverRule',
                'ExponentialMovingAverageCrossoverRule',
                'RelativeStrengthIndexTrasholdRule',
                'TripleExponentialDirectionChangeRule',
                'IchimokuKinkoHyoTenkanKijunCrossoverRule',
                'IchimokuKinkoHyoSenkouASenkouBCrossoverRule',
                'IchimokuKinkoHyoChikouCrossoverRule',
                'BollingerBandsLowerUpperCrossoverRule',
                'BollingerBandsLowerMidCrossoverRule',
                'BollingerBandsUpperMidCrossoverRule',
                'MovingAverageConvergenceDivergenceSignalLineCrossoverRule',
            ]
        else:
            print("No rule set specified, using the default - MovingAverageCrossoverRule")
            c.rules = ['MovingAverageCrossoverRule']

        if args.timeframe:
            c.timeframes = [args.timeframe]
        elif args.best_timeframes:
            c.timeframes = ['1h', '4h', '1d', '5m']
        else:
            print("No timeframe specified, using the default - 4h")
            c.timeframes = ['4h']
        
        return c

def constract_searchspace():
    """Writes json file with parameters searchspace.

    Structure:
        {
            rule: {
                'parameters': {attribute: search_space}
                'indicators': [
                    {
                        'name': string,
                        'parameters': {attribute: search_space}
                    }
                ]
            }
        }
    """

    def get_logspace(first, last, num, dtype=int):
        start = np.log10(first)
        stop = np.log10(last)
        space = list(sorted(set(np.logspace(start, stop, num, dtype=dtype))))
        return list(map(dtype, space))

    nested_dict = lambda: collections.defaultdict(nested_dict)
    data = collections.defaultdict(nested_dict)

    ranges = {
        '1m': get_logspace(10, 360, 7),
        '5m': get_logspace(8, 288, 7),
        '15m': get_logspace(8, 192, 7),
        '30m': get_logspace(8, 336, 7),
        '1h': get_logspace(6, 168, 7),
        '2h': get_logspace(6, 336, 7),
        '4h': get_logspace(6, 180, 7),
        '6h': get_logspace(6, 120, 7),
        '8h': get_logspace(6, 90, 7),
        '12h': get_logspace(6, 180, 7),
        '1d': get_logspace(7, 365, 25),
    }
    patience = get_logspace(1, 50, 5)

    indicator_names = [
        'PriceIndicator',
        'MovingAverageIndicator',
        'ExponentialMovingAverageIndicator',
        'RelativeStrengthIndexIndicator',
        'TripleExponentialIndicator',
        'IchimokuKinkoHyoIndicator',
        'BollingerBandsIndicator',
        'MovingAverageConvergenceDivergenceIndicator',
    ]

    rule_names = [
        'MovingAverageCrossoverRule',
        'ExponentialMovingAverageCrossoverRule',
        'RelativeStrengthIndexTrasholdRule',
        'TripleExponentialDirectionChangeRule',
        'IchimokuKinkoHyoTenkanKijunCrossoverRule',
        'IchimokuKinkoHyoSenkouASenkouBCrossoverRule',
        'IchimokuKinkoHyoChikouCrossoverRule',
        'IchimokuKinkoHyoSenkouASenkouBSupportResistanceRule',
        'BollingerBandsLowerMidCrossoverRule',
        'BollingerBandsUpperMidCrossoverRule',
        'BollingerBandsLowerUpperCrossoverRule',
        'MovingAverageConvergenceDivergenceSignalLineCrossoverRule',
    ]

    rule_indicators  = {
        'MovingAverageCrossoverRule': ['MovingAverageIndicator'] * 2,
        'ExponentialMovingAverageCrossoverRule': ['ExponentialMovingAverageIndicator'] * 2,
        'RelativeStrengthIndexTrasholdRule': ['RelativeStrengthIndexIndicator'],
        'TripleExponentialDirectionChangeRule': ['TripleExponentialIndicator'],
        'IchimokuKinkoHyoTenkanKijunCrossoverRule': ['IchimokuKinkoHyoIndicator'],
        'IchimokuKinkoHyoSenkouASenkouBCrossoverRule': ['IchimokuKinkoHyoIndicator'],
        'IchimokuKinkoHyoChikouCrossoverRule': ['IchimokuKinkoHyoIndicator', 'PriceIndicator'],
        'IchimokuKinkoHyoSenkouASenkouBSupportResistanceRule': ['IchimokuKinkoHyoIndicator', 'PriceIndicator'],
        'BollingerBandsLowerMidCrossoverRule': ['BollingerBandsIndicator', 'PriceIndicator'],
        'BollingerBandsUpperMidCrossoverRule': ['BollingerBandsIndicator', 'PriceIndicator'],
        'BollingerBandsLowerUpperCrossoverRule': ['BollingerBandsIndicator', 'PriceIndicator'],
        'MovingAverageConvergenceDivergenceSignalLineCrossoverRule': ['MovingAverageConvergenceDivergenceIndicator'],
    }

    for timeframe in ranges:

        space = ranges[timeframe]
        timeframe = data[timeframe]

        rule_parameters = {rule: {'patience': patience} for rule in rule_names}
        rule_parameters['RelativeStrengthIndexTrasholdRule'] |=  {
            'lower': list(range(20, 45, 5)),
            'upper': list(range(60, 85, 5)),
        }

        indicator_parameters = {indicator: {'length': space} for indicator in indicator_names}
        indicator_parameters['PriceIndicator'] = {}
        indicator_parameters['IchimokuKinkoHyoIndicator'] = {'short': space, 'long': space}
        indicator_parameters['BollingerBandsIndicator'] = {'length': space, 'mult': get_logspace(1.5, 3, 5, float)}
        indicator_parameters['MovingAverageConvergenceDivergenceIndicator'] = {'long': space, 'signal': space}

        for rule in rule_names:
            inds = []
            for indicator in rule_indicators[rule]:
                inds.append({'name': indicator, 'parameters': indicator_parameters[indicator]})
            timeframe[rule] = {'parameters': rule_parameters[rule], 'indicators': inds}

    return data

def get_experts_from_searchspace(timeframe: str,
                                 rule_name: str,
                                 cfg: str = 'searchspace.json') -> list[experts.RuleExpert]:
        cfg = json.load(open(cfg, 'r'))

        rule_parameters = cfg[timeframe][rule_name]['parameters']
        indicators_lst = cfg[timeframe][rule_name]['indicators']
        indicator_cls_names = list(ind['name'] for ind in indicators_lst)
        indicator_parameters = [ind['parameters'] for ind in indicators_lst]

        res = []
        for rule_params in product(*rule_parameters.values()):
            rule_kwargs = {key: val for key, val in zip(list(rule_parameters), rule_params)}

            indicator_combinations = [product(*ind.values()) for ind in indicator_parameters]
            for inds_params in product(*indicator_combinations):
                lst = []
                for cls_name, (attrs, params) in zip(indicator_cls_names,
                                                     zip((param.keys() for param in indicator_parameters), inds_params)):
                    indicator_kwargs = {attr: val for attr, val in zip(attrs, params)}
                    try:
                        ind = getattr(indicators, cls_name)(**indicator_kwargs)
                        lst.append(ind)
                    except ValueError as err:
                        break

                else:
                    rule = getattr(rules, rule_name)(**rule_kwargs)
                    try:
                        res.append(experts.RuleExpert(rule, lst))
                    except ValueError as err:
                        pass

        return res

def serialize_expert_to_json(filename: str = 'expert.json',
                             expert: experts.BaseExpert = None):
    """
    Structure:
        {
            'name': string,
            'inner experts: [serialized experts]'
        }

        OR

        {
            'name': string,
            'rule': {
                'name': string,
                'parameters': {attribute: value}
            }
            'indicators': [
                {
                    'name': 'string',
                    'parameters': {attribute: value}
                    'estimation': {estimation: value}
                }
            ]
        }
    """

    def get_hierarchy(expert: experts.BaseExpert):
        state = {}
        state['name'] = expert.__class__.__name__
        state['parameters'] = expert.get_parameters()
        if hasattr(expert, '_inner_experts'):
            state['inner experts'] = [get_hierarchy(exp) for exp in expert._inner_experts]
        else:
            rule, indicators = expert._rule, expert._indicators
            rule = {'name': rule.__class__.__name__, 'parameters': rule.get_parameters()}
            indicators = [{'name': indicator.__class__.__name__,
                           'parameters': indicator.get_parameters()}
                                                    for indicator in indicators]
            state['parameters'] = {'rule': rule, 'indicators': indicators}
            state['estimation'] = expert.estimation

        return state

    hierarchy = get_hierarchy(expert)
    json.dump(hierarchy, open(filename, 'w'), indent=4)

def deserialize_expert_from_json(filename: str = 'expert.json'):
    def deserialize_expert_from_dict(hierarchy):
        if 'inner experts' in hierarchy:
            expert = getattr(experts, hierarchy['name'])(**hierarchy['parameters'])
            inner = [deserialize_expert_from_dict(exp) for exp in hierarchy['inner experts']]
            expert.set_experts(inner)
        else:
            rule = hierarchy['parameters']['rule']
            inds = hierarchy['parameters']['indicators']
            rule = getattr(rules, rule['name'])(**rule['parameters'])
            inds = [getattr(indicators, ind['name'])(**ind['parameters']) for ind in inds]
            expert = experts.RuleExpert(rule, inds)
            expert.estimation = hierarchy['estimation']

        return expert

    hierarchy = json.load(open(filename, 'r'))
    expert = deserialize_expert_from_dict(hierarchy)
    return expert



if __name__ == '__main__':
    print(Config.from_args().to_json())
    constract_searchspace()