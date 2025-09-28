class Portfolio:
    def __init__(self, initial_cash=1000000, commission=0.001, slippage = 0.0002, min_shares = 10):
        self.inital_cash = initial_cash
        self.cash = initial_cash
        self.positions = {} # symbol -> number of shares

        self.commission = commission
        self.slippage = slippage
        self.min_shares = min_shares

        self.trades = [] # list of trade records

        self.target_stocks = 7 


    def add_asset(self, asset):
        self.assets.append(asset)

    def total_value(self):
        return sum(asset.value for asset in self.assets)