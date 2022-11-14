from trainer import Trainer
from config import Config

if __name__ == "__main__":

    cfg = Config.from_args()
    trainer = Trainer()
    trader = trainer.construct_trader(cfg)
    trainer.simulate_pair_trader(trader, 1000, display=True)