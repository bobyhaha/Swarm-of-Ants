"""Integer society credits. Rights are not money and are never counted twice."""

from dataclasses import dataclass, field


def amount(value):
    if type(value) is not int or value < 0:
        raise ValueError("Amount must be a nonnegative integer")
    return value


@dataclass
class Ledger:
    initial: int
    emit: object
    balances: dict = field(init=False)
    burned: int = 0
    rewarded: int = 0
    settled: set = field(default_factory=set)

    def __post_init__(self):
        self.balances = {"bank": amount(self.initial)}

    @property
    def wealth(self):
        return sum(self.balances.values())

    def available(self, wallet):
        return self.balances.get(wallet, 0)

    def transfer(self, source, target, value, reason):
        amount(value)
        if value > self.available(source):
            raise ValueError("Insufficient wallet balance")
        self.balances[source] = self.available(source) - value
        self.balances[target] = self.available(target) + value
        self.emit("transfer", source=source, target=target, amount=value, reason=reason)
        self.check()

    def burn(self, wallet, value, reason):
        amount(value)
        if value > self.available(wallet):
            raise ValueError("Cannot burn unowned money")
        self.balances[wallet] -= value
        self.burned += value
        self.emit("burn", wallet=wallet, amount=value, reason=reason)
        self.check()

    def reward_once(self, task_id, value):
        amount(value)
        if task_id in self.settled:
            return False
        self.settled.add(task_id)
        self.balances["bank"] += value
        self.rewarded += value
        self.emit("reward", task_id=task_id, amount=value)
        self.check()
        return True

    def check(self):
        if any(type(x) is not int or x < 0 for x in self.balances.values()):
            raise AssertionError("Invalid balance")
        if self.wealth != self.initial + self.rewarded - self.burned:
            raise AssertionError("Money was created or lost outside the ledger")
