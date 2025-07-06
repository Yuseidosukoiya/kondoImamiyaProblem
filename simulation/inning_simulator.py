class InningSimulator:
    def __init__(self, team):
        self.team = team
        self.batter_index = 0  # 打順はゲーム全体で継続
        self.reset_inning()

    def reset_inning(self):
        self.bases = {'1st': None, '2nd': None, '3rd': None}
        self.outs = 0
        self.runs = 0

    def process_walk(self, batter):
        if self.bases['1st'] is None:
            self.bases['1st'] = batter
        else:
            displaced = self.bases['1st']
            self.bases['1st'] = batter
            if self.bases['2nd'] is None:
                self.bases['2nd'] = displaced
            else:
                displaced2 = self.bases['2nd']
                self.bases['2nd'] = displaced
                if self.bases['3rd'] is None:
                    self.bases['3rd'] = displaced2
                else:
                    self.runs += 1
                    self.bases['3rd'] = displaced2

    def process_single(self, batter):
        if self.bases['3rd'] is not None:
            self.runs += 1
            self.bases['3rd'] = None
        if self.bases['2nd'] is not None:
            if self.bases['2nd'].name == 'imamiya':
                self.runs += 1
            else:
                if self.bases['3rd'] is None:
                    self.bases['3rd'] = self.bases['2nd']
                else:
                    self.runs += 1
            self.bases['2nd'] = None
        if self.bases['1st'] is not None:
            if self.bases['2nd'] is None:
                self.bases['2nd'] = self.bases['1st']
            else:
                if self.bases['3rd'] is None:
                    self.bases['3rd'] = self.bases['1st']
                else:
                    self.runs += 1
            self.bases['1st'] = None
        self.bases['1st'] = batter

    def process_double(self, batter):
        for base in ['3rd', '2nd']:
            if self.bases[base] is not None:
                self.runs += 1
                self.bases[base] = None
        if self.bases['1st'] is not None:
            self.bases['3rd'] = self.bases['1st']
            self.bases['1st'] = None
        self.bases['2nd'] = batter

    def process_triple(self, batter):
        for base in ['1st', '2nd', '3rd']:
            if self.bases[base] is not None:
                self.runs += 1
                self.bases[base] = None
        self.bases['3rd'] = batter

    def process_hr(self, batter):
        for base in ['1st', '2nd', '3rd']:
            if self.bases[base] is not None:
                self.runs += 1
                self.bases[base] = None
        self.runs += 1

    def simulate_inning(self):
        self.reset_inning()
        while self.outs < 3:
            batter = self.team.lineup[self.batter_index % len(self.team.lineup)]
            outcome = batter.plate_appearance()
            if outcome == 'out':
                self.outs += 1
            elif outcome == 'walk':
                self.process_walk(batter)
            elif outcome == 'single':
                self.process_single(batter)
            elif outcome == 'double':
                self.process_double(batter)
            elif outcome == 'triple':
                self.process_triple(batter)
            elif outcome == 'hr':
                self.process_hr(batter)
            self.batter_index += 1
        return self.runs