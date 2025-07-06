import numpy as np

class Player:
    def __init__(self, name, stats):
        """
        :param name: 選手名（例: 'imamiya', 'kondo'）
        :param stats: 辞書形式の成績データ（打席、安打、二塁打、…）
        """
        self.name = name
        self.stats = stats
        self.probabilities = self.calc_probabilities()
        
    def calc_probabilities(self):
        singles = self.stats['安打'] - (self.stats['二塁打'] + self.stats['三塁打'] + self.stats['本塁打'])
        p_single = singles / self.stats['打席']
        p_double = self.stats['二塁打'] / self.stats['打席']
        p_triple = self.stats['三塁打'] / self.stats['打席']
        p_hr = self.stats['本塁打'] / self.stats['打席']
        p_walk = (self.stats['四球'] + self.stats['故意四'] + self.stats['死球']) / self.stats['打席']
        p_out = 1 - (p_single + p_double + p_triple + p_hr + p_walk)
        return {
            'single': p_single,
            'double': p_double,
            'triple': p_triple,
            'hr': p_hr,
            'walk': p_walk,
            'out': p_out
        }
    
    def plate_appearance(self):
        outcomes = ['single', 'double', 'triple', 'hr', 'walk', 'out']
        probs = [self.probabilities[outcome] for outcome in outcomes]
        return np.random.choice(outcomes, p=probs)