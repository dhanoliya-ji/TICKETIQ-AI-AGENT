"""Triage one ticket:  python -m ticketiq --subject "Charged twice" --body "..." --tier pro"""
import argparse
import json

from .triage import TicketIQ

ap = argparse.ArgumentParser(prog="python -m ticketiq")
ap.add_argument("--subject", required=True)
ap.add_argument("--body", required=True)
ap.add_argument("--tier", default="pro", choices=["free", "pro", "enterprise"])
args = ap.parse_args()
print(json.dumps(TicketIQ().handle(args.subject, args.body, args.tier), indent=2))
