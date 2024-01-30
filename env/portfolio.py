class Portfolio:
    def __init__(self, asset, fiat, interest_asset = 0, interest_fiat = 0):
        self.asset =asset
        self.fiat =fiat
        self.interest_asset = interest_asset
        self.interest_fiat = interest_fiat
    
    #Calculate the sum of asset factor * new "close" price + FIAT -Interest asset + price - Interest FIAT 
    def valorisation(self, price):
        return sum([
            self.asset * price,
            self.fiat,
            - self.interest_asset * price,
            - self.interest_fiat
        ])
    # if FIAT and interest asset = 0 real position =1    . I dont't know what this doing.
    def real_position(self, price):
        return (self.asset - self.interest_asset)* price / self.valorisation(price)
    
    def position(self, price):
        return self.asset * price / self.valorisation(price)

    def fiat_value(self):
        return self.fiat
        
    def trade_to_position(self, position, price, trading_fees):
        # Repay interest
        current_position = self.position(price) #if Fiat and interest =0 then current_position =1
        interest_reduction_ratio = 1
        
        # check to reduce interest. why?, I don't know
        if (position <= 0 and current_position < 0):
            interest_reduction_ratio = min(1, position/current_position)
        elif (position >= 1 and current_position > 1):
            interest_reduction_ratio = min(1, (position-1)/(current_position-1))
        #Interest reduction if "interest_reduction_ratio" < 1
        if interest_reduction_ratio < 1:
            self.asset = self.asset - (1-interest_reduction_ratio) * self.interest_asset
            self.fiat = self.fiat - (1-interest_reduction_ratio) * self.interest_fiat
            self.interest_asset = interest_reduction_ratio * self.interest_asset
            self.interest_fiat = interest_reduction_ratio * self.interest_fiat
        # Proceed to trade
        
        asset_trade = (position * self.valorisation(price) / price - self.asset)
        if asset_trade > 0:
            asset_trade = asset_trade / (1 - trading_fees + (trading_fees * position))
            asset_fiat = - asset_trade * price
            self.asset = self.asset + asset_trade * (1 - trading_fees)
            self.fiat = self.fiat + asset_fiat
        else:
            asset_trade = asset_trade / (1 - trading_fees * position)
            asset_fiat = - asset_trade * price
            self.asset = self.asset + asset_trade 
            self.fiat = self.fiat + asset_fiat * (1 - trading_fees)
            
    # def update_interest(self, borrow_interest_rate):
    #     self.interest_asset = max(0, - self.asset)*borrow_interest_rate
    #     self.interest_fiat = max(0, - self.fiat)*borrow_interest_rate
        
    def __str__(self): return f"{self.__class__.__name__}({self.__dict__})"
    
    def describe(self, price): print("Value : ", self.valorisation(price), "Position : ", self.position(price))
    
    def get_portfolio_distribution(self):
        return {
            "asset":max(0, self.asset),
            "fiat":max(0, self.fiat),
            "borrowed_asset":max(0, -self.asset),
            "borrowed_fiat":max(0, -self.fiat),
            "interest_asset":self.interest_asset,
            "interest_fiat":self.interest_fiat,
        }

class TargetPortfolio(Portfolio):
    def __init__(self, position ,value, price):
        super().__init__(
            #Calculate asset factor position * initial value portfolio / first "close" price: 1*100/200=0.5 factor
            asset = position * value / price,
            #Calculate FIAT oposite position (1-position) * Initial value: (1-0)*100=100
            fiat = (1-position) * value,
            #interest
            interest_asset = 0,
            interest_fiat = 0
        )
