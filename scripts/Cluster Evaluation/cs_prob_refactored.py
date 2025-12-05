"""
Main entry point for CS-PROB evaluation system.
Refactored modular version with clean architecture.
"""
#!/usr/bin/env python3

import argparse
import time
from pathlib import Path
from typing import Dict

import pandas as pd

from config import (
    JUDGE_API_KEYS,
    MODELS_UNDER_TEST,
    SYSTEM_PROMPT,
    TEMPERATURE,
    REQUEST_TIMEOUT,
    DEFAULT_QUESTION_COLUMN,
    DEFAULT_REFERENCE_COLUMN,
    FALLBACK_REFERENCE_COLUMN,
    OUTPUT_DIRECTORY,
    DEFAULT_OUTPUT_FILENAME,
    AGGREGATED_CSV_FILENAME
)
from excel_reader import ExcelReader
from llm_client import LLMClient
from judge_client import JudgeClient
from evaluator import SheetEvaluator


class CSProbApplication:
    """Main application class for CS-PROB evaluation system."""
    
    def __init__(self, args: argparse.Namespace):
        """
        Initialize application with command-line arguments.
        
        Args:
            args: Parsed command-line arguments
        """
        self.args = args
        self.input_path = Path(args.input_xlsx)
        self.output_path = Path(args.output_xlsx)
        self.output_dir = Path(OUTPUT_DIRECTORY)
        
        # Validate configuration
        self._validate_configuration()
        
        # Initialize components
        self.excel_reader = ExcelReader(self.input_path, args.question_col)
        self.llm_clients = self._create_llm_clients()
        self.judge_client = JudgeClient(JUDGE_API_KEYS)
    
    def _validate_configuration(self):
        """Validate that configuration is complete."""
        # Check API keys are configured
        placeholder_pattern = "PASTE_YOUR_KEY_"
        if any(placeholder_pattern in key for key in JUDGE_API_KEYS):
            print("\n" + "="*80)
            print("ERROR: Please configure JUDGE_API_KEYS in config.py")
            print("="*80 + "\n")
            raise SystemExit(1)
        
        # Check input file exists
        if not self.input_path.exists():
            print(f"\n[ERROR] Input file not found: {self.input_path}\n")
            raise SystemExit(1)
    
    def _create_llm_clients(self) -> Dict[str, LLMClient]:
        """Create LLM client instances for all models under test."""
        clients = {}
        for model_name, endpoint_url in MODELS_UNDER_TEST.items():
            clients[model_name] = LLMClient(
                model_name=model_name,
                endpoint_url=endpoint_url,
                system_prompt=SYSTEM_PROMPT,
                temperature=TEMPERATURE,
                timeout=self.args.timeout
            )
        return clients
    
    def run(self):
        """Execute the main evaluation workflow."""
        self._print_startup_banner()
        
        # Read Excel workbook
        sheets_data = self.excel_reader.read_workbook()
        
        # Process each sheet
        start_time = time.time()
        evaluated_sheets = self._evaluate_all_sheets(sheets_data)
        total_time = time.time() - start_time
        
        # Save results
        self._save_results(evaluated_sheets, sheets_data)
        
        # Print completion message
        self._print_completion_message(total_time)
    
    def _print_startup_banner(self):
        """Print application startup information."""
        print("\n" + "="*80)
        print("CS-PROB: University CS Exam LLM Evaluation")
        print("="*80)
        print(f"Input file       : {self.input_path}")
        print(f"Output file      : {self.output_path}")
        print(f"Output directory : {self.output_dir.absolute()}")
        print(f"Timeout          : {self.args.timeout}s")
        print(f"Question column  : {self.args.question_col}")
        print(f"Reference column : {self.args.reference_col}")
        print(f"System prompt    : {SYSTEM_PROMPT}")
        print(f"Temperature      : {TEMPERATURE}")
        print(f"Judge keys       : {len(JUDGE_API_KEYS)} keys configured")
        print(f"Models under test: {len(MODELS_UNDER_TEST)}")
        for model_name in MODELS_UNDER_TEST.keys():
            print(f"  - {model_name}")
        print("="*80)
    
    def _evaluate_all_sheets(
        self,
        sheets_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, pd.DataFrame]:
        """
        Evaluate all sheets in the workbook.
        
        Args:
            sheets_data: Dictionary mapping sheet names to DataFrames
            
        Returns:
            Dictionary mapping sheet names to result DataFrames
        """
        evaluated_sheets = {}
        
        for sheet_name, dataframe in sheets_data.items():
            # Determine reference column name
            reference_column = self._get_reference_column_name(
                dataframe,
                sheet_name
            )
            
            # Create evaluator and process sheet
            sheet_start_time = time.time()
            evaluator = SheetEvaluator(
                sheet_name=sheet_name,
                dataframe=dataframe,
                llm_clients=self.llm_clients,
                judge_client=self.judge_client,
                question_column=self.args.question_col,
                reference_column=reference_column
            )
            
            result_dataframe = evaluator.evaluate_all()
            sheet_elapsed = time.time() - sheet_start_time
            
            # Save per-sheet result
            sheet_output_path = self.output_dir / f"{sheet_name}_results.xlsx"
            result_dataframe.to_excel(sheet_output_path, index=False)
            print(f"\n[✓] Saved sheet results: {sheet_output_path}")
            print(f"    Sheet processing time: {sheet_elapsed:.1f}s")
            
            evaluated_sheets[sheet_name] = result_dataframe
        
        return evaluated_sheets
    
    def _get_reference_column_name(
        self,
        dataframe: pd.DataFrame,
        sheet_name: str
    ) -> str:
        """
        Determine the correct reference column name.
        
        Args:
            dataframe: Sheet DataFrame
            sheet_name: Sheet name for error messages
            
        Returns:
            Reference column name
        """
        # Try primary column name
        if self.args.reference_col in dataframe.columns:
            return self.args.reference_col
        
        # Try fallback
        if FALLBACK_REFERENCE_COLUMN in dataframe.columns:
            return FALLBACK_REFERENCE_COLUMN
        
        # Not found
        available_columns = list(dataframe.columns)
        raise ValueError(
            f"Reference column '{self.args.reference_col}' not found in "
            f"sheet '{sheet_name}'. Available columns: {available_columns}"
        )
    
    def _save_results(
        self,
        evaluated_sheets: Dict[str, pd.DataFrame],
        original_sheets: Dict[str, pd.DataFrame]
    ):
        """
        Save aggregated and consolidated results.
        
        Args:
            evaluated_sheets: Evaluated result DataFrames
            original_sheets: Original DataFrames (for structure)
        """
        if not evaluated_sheets:
            return
        
        # Save aggregated CSV
        aggregated_dataframes = [
            df.assign(__sheet=sheet_name)
            for sheet_name, df in evaluated_sheets.items()
        ]
        aggregated_df = pd.concat(aggregated_dataframes, ignore_index=True)
        
        aggregated_csv_path = self.output_dir / AGGREGATED_CSV_FILENAME
        aggregated_df.to_csv(aggregated_csv_path, index=False)
        print(f"\n[✓] Saved aggregated CSV: {aggregated_csv_path}")
        
        # Save consolidated workbook
        with pd.ExcelWriter(self.output_path, engine='openpyxl') as writer:
            for sheet_name in original_sheets.keys():
                if sheet_name in evaluated_sheets:
                    # Use evaluated data
                    final_df = evaluated_sheets[sheet_name]
                else:
                    # Fallback to original
                    final_df = original_sheets[sheet_name]
                
                final_df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        print(f"[✓] Saved consolidated workbook: {self.output_path}")
    
    def _print_completion_message(self, total_time: float):
        """Print completion message with timing."""
        print(f"\n{'='*80}")
        print(f"COMPLETED")
        print(f"Total processing time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
        print(f"{'='*80}\n")


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="CS-PROB: Evaluate LLMs on university CS exam questions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python3 cs_prob.py --input_xlsx exams.xlsx
  python3 cs_prob.py --input_xlsx data/questions.xlsx --output_xlsx results.xlsx
  python3 cs_prob.py --input_xlsx exams.xlsx --question_col "Problem"

Note: Configure JUDGE_API_KEYS in config.py before running.
        """
    )
    
    parser.add_argument(
        "--input_xlsx",
        type=str,
        required=True,
        help="Path to Excel workbook with multiple sheets"
    )
    
    parser.add_argument(
        "--output_xlsx",
        type=str,
        default=DEFAULT_OUTPUT_FILENAME,
        help=f"Output Excel workbook with added columns (default: {DEFAULT_OUTPUT_FILENAME})"
    )
    
    parser.add_argument(
        "--timeout",
        type=int,
        default=REQUEST_TIMEOUT,
        help=f"HTTP timeout seconds (default: {REQUEST_TIMEOUT})"
    )
    
    parser.add_argument(
        "--question_col",
        type=str,
        default=DEFAULT_QUESTION_COLUMN,
        help=f"Column name for question text (default: {DEFAULT_QUESTION_COLUMN})"
    )
    
    parser.add_argument(
        "--reference_col",
        type=str,
        default=DEFAULT_REFERENCE_COLUMN,
        help=f"Column name for reference answer (default: {DEFAULT_REFERENCE_COLUMN})"
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_arguments()
    app = CSProbApplication(args)
    app.run()


if __name__ == "__main__":
    main()
