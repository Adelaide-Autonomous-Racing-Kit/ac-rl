import argparse

from acrl.agent import SACAgent


def main():
    args = parse_arguments()
    agent = SACAgent(args.config)
    agent.run()


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to configuration")
    return parser.parse_args()


if __name__ == "__main__":
    main()
