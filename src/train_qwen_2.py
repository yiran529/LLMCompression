try:
    from .config import ExperimentConfig
    from .trainer import train
except ImportError:
    from config import ExperimentConfig
    from trainer import train


def main() -> None:
    cfg = ExperimentConfig()
    train(cfg)


if __name__ == "__main__":
    main()

