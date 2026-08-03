#!/usr/bin/env python3
import sys
import argparse
import os

def main():
    parser = argparse.ArgumentParser(
        description="MedSAM2 Preprocessing Router",
        usage="python run_preprocessing.py <command> [<args>]"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # RatData Subparser
    parser_rat = subparsers.add_parser("rat", help="Preprocess RatData")
    
    # HumanData Subparser
    parser_human = subparsers.add_parser("human", help="Preprocess HumanData")
    
    # DIAS Data Subparser
    parser_dias = subparsers.add_parser("dias", help="Preprocess DIAS data")
    
    # Discover Window Subparser
    parser_discover = subparsers.add_parser("discover-window", help="Discover optimal Window Level values")
    
    # NIfTI Pipeline Subparsers
    parser_dicom2nifti = subparsers.add_parser("dicom2nifti", help="Convert DICOMs to raw NIfTI")
    parser_gen_mask = subparsers.add_parser("generate-mask", help="Generate hessian masks from raw NIfTI")
    parser_prep_nifti = subparsers.add_parser("preprocess-nifti", help="Preprocess NIfTI images")
    
    # If no arguments are provided
    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(1)
        
    args, unknown_args = parser.parse_known_args()
    
    # Route to the appropriate module
    if args.command == "rat":
        from preprocess import ratdata
        sys.argv = [sys.argv[0]] + unknown_args
        ratdata.preprocess_ratdata(ratdata.parser.parse_args())
    elif args.command == "human":
        from preprocess import humandata
        sys.argv = [sys.argv[0]] + unknown_args
        humandata.preprocess_humandata(humandata.parser.parse_args())
    elif args.command == "dias":
        from preprocess import dias_data
        sys.argv = [sys.argv[0]] + unknown_args
        parsed_args = dias_data.parser.parse_args()
        dias_data.preprocess_dias_for_medsam2(parsed_args.base_data_dir, parsed_args.output_dir, label_type="standard", args=parsed_args)
        dias_data.preprocess_dias_for_medsam2(parsed_args.base_data_dir, parsed_args.output_dir, label_type="SALE", args=parsed_args)
        dias_data.preprocess_dias_for_medsam2(parsed_args.base_data_dir, parsed_args.output_dir, label_type="RDFA", args=parsed_args)
    elif args.command == "discover-window":
        from utils import window_level
        sys.argv = [sys.argv[0]] + unknown_args
        parsed = window_level.parser.parse_args()
        if os.path.exists(os.path.expanduser(parsed.human_dir)):
            window_level.analyze_dataset(os.path.expanduser(parsed.human_dir))
        if os.path.exists(os.path.expanduser(parsed.rat_dir)):
            window_level.analyze_dataset(os.path.expanduser(parsed.rat_dir))
        if os.path.exists(os.path.expanduser(parsed.dias_dir)):
            window_level.analyze_dataset(os.path.expanduser(parsed.dias_dir))
    elif args.command == "dicom2nifti":
        from preprocess import dicom_to_nifti
        sys.argv = [sys.argv[0]] + unknown_args
        dicom_to_nifti.preprocess_dicom_to_nifti(dicom_to_nifti.parser.parse_args())
    elif args.command == "generate-mask":
        from preprocess import generate_mask
        sys.argv = [sys.argv[0]] + unknown_args
        generate_mask.generate_masks(generate_mask.parser.parse_args())
    elif args.command == "preprocess-nifti":
        from preprocess import preprocess_nifti
        sys.argv = [sys.argv[0]] + unknown_args
        preprocess_nifti.preprocess_nifti(preprocess_nifti.parser.parse_args())
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
