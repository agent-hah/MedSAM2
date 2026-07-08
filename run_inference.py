#!/usr/bin/env python3
import sys
import argparse

def main():
    parser = argparse.ArgumentParser(
        description="MedSAM2 Inference Router",
        usage="python run_inference.py <command> [<args>]"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    parser_rat_dias = subparsers.add_parser("rat-dias", help="Run RatData DIAS Inference")
    parser_human_dias = subparsers.add_parser("human-dias", help="Run HumanData DIAS Inference")
    parser_rat = subparsers.add_parser("rat", help="Run standard RatData Inference")
    parser_human = subparsers.add_parser("human", help="Run standard HumanData Inference")
    
    # If no arguments are provided
    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(1)
        
    args, unknown_args = parser.parse_known_args()
    
    # Route to the appropriate module
    if args.command == "rat-dias":
        from inference import infer_ratdata_dias
        sys.argv = [sys.argv[0]] + unknown_args
        infer_ratdata_dias.main()
    elif args.command == "human-dias":
        from inference import infer_humandata_dias
        sys.argv = [sys.argv[0]] + unknown_args
        infer_humandata_dias.main()
    elif args.command == "rat":
        from inference import infer_ratdata
        sys.argv = [sys.argv[0]] + unknown_args
        infer_ratdata.main()
    elif args.command == "human":
        from inference import infer_humandata
        sys.argv = [sys.argv[0]] + unknown_args
        infer_humandata.main()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
