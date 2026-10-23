import argparse
from pathlib import Path

from telco_churn.dataset import write_report

parser = argparse.ArgumentParser()
parser.add_argument("--source", type=Path, default=Path("data/raw/telco.csv"))
parser.add_argument("--output", type=Path, default=Path("reports/data_overview.json"))
args = parser.parse_args()

write_report(args.source, args.output)
print(f"Report written to {args.output}")