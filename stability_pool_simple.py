import math


class stability_pool:
    last_recovery_time = 0
    current_balance = 0
    time_liquidation = 0

    def __init__(self, initial_balance, recovery_interval, recovery_volume, **kwargs):
        initial_share_institutional = kwargs.get('share_institutional', 1)
        self.initial_balance = initial_balance
        self.initial_balance_institutional = initial_balance * initial_share_institutional
        self.initial_balance_retail = max(self.initial_balance - self.initial_balance_institutional, 0)

        self.current_balance = initial_balance
        self.current_balance_institutional = self.initial_balance_institutional
        self.current_balance_retail = self.initial_balance_retail

        self.recovery_interval = recovery_interval
        self.recovery_volume = recovery_volume

        self.recovery_halflife_retail = kwargs.get('recovery_halflife_retail', recovery_interval * 100)

    def do_tick(self, time, max_recovery):
        current_recovery_volume_retail = 0
        current_recovery_volume_institutional = 0

        missing_volume_retail = self.initial_balance_retail - self.current_balance_retail
        if missing_volume_retail > 0:
            if self.recovery_halflife_retail == 0:
                current_recovery_volume_retail = min(missing_volume_retail, max_recovery)
                self.current_balance_retail += current_recovery_volume_retail
            else:
                next_missing_volume_retail = missing_volume_retail * pow(0.5, 1 / (self.recovery_halflife_retail * 24 * 60))
                current_recovery_volume_retail = missing_volume_retail - next_missing_volume_retail
                current_recovery_volume_retail = min(current_recovery_volume_retail, max_recovery)
                self.current_balance_retail += current_recovery_volume_retail

        more_to_recovery = max_recovery - current_recovery_volume_retail
        if more_to_recovery > 0 and (time - self.last_recovery_time) / 1_000_000 > self.recovery_interval:
            missing_volume_institutional = self.initial_balance_institutional - self.current_balance_institutional
            current_recovery_volume_institutional = min(more_to_recovery, missing_volume_institutional)
            self.current_balance_institutional += current_recovery_volume_institutional
            self.last_recovery_time = time

        self.current_balance = self.current_balance_retail + self.current_balance_institutional
        to_return = current_recovery_volume_retail + current_recovery_volume_institutional
        return to_return

    def do_check_liquidation_size(self):
        v = self.current_balance
        return v

    def do_set_liquidation_size(self, size):
        share_institutional = self.current_balance_institutional / self.current_balance
        self.current_balance_institutional -= size * share_institutional
        self.current_balance_retail -= size * (1 - share_institutional)
        self.current_balance -= size