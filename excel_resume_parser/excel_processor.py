"""
Excel Data Processor

This module handles preprocessing of Excel/XLSX files for resume parsing.
It removes duplicate headers and converts data into structured dictionaries.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import io
import os
from core.custom_logger import CustomLogger

# Initialize logger
logger_manager = CustomLogger()
logger = logger_manager.get_logger("excel_processor")


class ExcelProcessor:
    """
    Processes Excel files for resume data extraction.
    Handles duplicate headers, data cleaning, and conversion to dictionaries.
    """

    def __init__(self):
        """Initialize the Excel processor."""
        self.supported_extensions = [".xlsx", ".xls", ".xlsm"]

    def validate_excel_file(self, file_path: str) -> bool:
        """
        Validate if the file is a supported Excel format.

        Args:
            file_path: Path to the Excel file

        Returns:
            bool: True if valid Excel file, False otherwise
        """
        try:
            file_ext = Path(file_path).suffix.lower()
            if file_ext not in self.supported_extensions:
                logger.error(f"Unsupported file extension: {file_ext}")
                return False

            # Try to read the file to check if it's valid
            pd.read_excel(file_path, nrows=1)
            return True

        except Exception as e:
            logger.error(f"Excel file validation failed: {e}")
            return False

    def detect_and_remove_duplicate_headers(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Detect and remove duplicate headers, keeping only the first occurrence.

        Args:
            df: Input DataFrame

        Returns:
            Tuple of (cleaned_dataframe, list_of_removed_headers)
        """
        try:
            original_columns = list(df.columns)
            logger.info(f"Original columns: {original_columns}")

            # Find duplicate column names (case-insensitive)
            seen_columns = {}
            columns_to_keep = []
            duplicate_columns = []

            for i, col in enumerate(original_columns):
                col_lower = str(col).lower().strip()

                if col_lower in seen_columns:
                    # This is a duplicate column
                    duplicate_columns.append(col)
                    logger.warning(f"Duplicate column found: '{col}' at index {i}")
                else:
                    # First occurrence of this column
                    seen_columns[col_lower] = i
                    columns_to_keep.append(i)

            # Keep only the first occurrence of each column
            if duplicate_columns:
                df_cleaned = df.iloc[:, columns_to_keep]
                logger.info(
                    f"Removed {len(duplicate_columns)} duplicate columns: {duplicate_columns}"
                )
            else:
                df_cleaned = df.copy()
                logger.info("No duplicate columns found")

            return df_cleaned, duplicate_columns

        except Exception as e:
            logger.error(f"Error removing duplicate headers: {e}")
            return df, []

    def clean_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and standardize column names.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with cleaned column names
        """
        try:
            original_columns = list(df.columns)

            # Clean column names
            cleaned_columns = []
            for col in original_columns:
                cleaned_col = str(col).strip().lower()
                # Remove special characters and replace spaces with underscores
                cleaned_col = "".join(
                    c if c.isalnum() or c == "_" else "_" for c in cleaned_col
                )
                # Remove consecutive underscores
                cleaned_col = "_".join(filter(None, cleaned_col.split("_")))
                cleaned_columns.append(cleaned_col)

            df.columns = cleaned_columns
            logger.info(
                f"Column names cleaned: {dict(zip(original_columns, cleaned_columns))}"
            )

            return df

        except Exception as e:
            logger.error(f"Error cleaning column names: {e}")
            return df

    def convert_rows_to_dictionaries(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Convert DataFrame rows to list of dictionaries.

        Args:
            df: Input DataFrame

        Returns:
            List of dictionaries representing each row
        """
        try:
            # Replace NaN values with None
            df_filled = df.where(pd.notnull(df), None)

            # Convert to list of dictionaries
            row_dicts = df_filled.to_dict("records")

            # Clean up empty or all-None rows
            cleaned_rows = []
            for i, row_dict in enumerate(row_dicts):
                # Check if row has any non-None values
                if any(
                    value is not None and str(value).strip() != ""
                    for value in row_dict.values()
                ):
                    cleaned_rows.append(row_dict)
                else:
                    logger.debug(f"Skipping empty row at index {i}")

            logger.info(f"Converted {len(cleaned_rows)} valid rows to dictionaries")
            return cleaned_rows

        except Exception as e:
            logger.error(f"Error converting rows to dictionaries: {e}")
            return []

    def process_excel_file(
        self, file_path: str, sheet_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Main method to process an Excel file and return structured data.

        Args:
            file_path: Path to the Excel file
            sheet_name: Specific sheet name to process (None for first sheet)

        Returns:
            List of dictionaries representing the Excel data
        """
        try:
            logger.info(f"Processing Excel file: {file_path}")

            # Validate file
            if not self.validate_excel_file(file_path):
                raise ValueError(f"Invalid Excel file: {file_path}")

            # Read Excel file
            if sheet_name:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                logger.info(f"Reading sheet: {sheet_name}")
            else:
                df = pd.read_excel(file_path)
                logger.info("Reading first sheet")

            logger.info(f"Original DataFrame shape: {df.shape}")

            # Remove duplicate headers
            df_no_duplicates, removed_headers = (
                self.detect_and_remove_duplicate_headers(df)
            )

            # Clean column names
            df_cleaned = self.clean_column_names(df_no_duplicates)

            # Convert to dictionaries
            row_dicts = self.convert_rows_to_dictionaries(df_cleaned)

            logger.info(
                f"Successfully processed {len(row_dicts)} records from Excel file"
            )
            return row_dicts

        except Exception as e:
            logger.error(f"Error processing Excel file {file_path}: {e}")
            raise

    def process_excel_bytes(
        self, file_bytes: bytes, filename: str, sheet_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Process Excel file from bytes data.

        Args:
            file_bytes: Excel file as bytes
            filename: Original filename for logging
            sheet_name: Specific sheet name to process (None for first sheet)

        Returns:
            List of dictionaries representing the Excel data
        """
        try:
            logger.info(f"Processing Excel bytes for file: {filename}")

            # Create BytesIO object
            excel_buffer = io.BytesIO(file_bytes)

            # Read Excel file from bytes
            if sheet_name:
                df = pd.read_excel(excel_buffer, sheet_name=sheet_name)
                logger.info(f"Reading sheet: {sheet_name}")
            else:
                df = pd.read_excel(excel_buffer)
                logger.info("Reading first sheet")

            logger.info(f"Original DataFrame shape: {df.shape}")

            # Remove duplicate headers
            df_no_duplicates, removed_headers = (
                self.detect_and_remove_duplicate_headers(df)
            )

            # Clean column names
            df_cleaned = self.clean_column_names(df_no_duplicates)

            # Convert to dictionaries
            row_dicts = self.convert_rows_to_dictionaries(df_cleaned)

            logger.info(
                f"Successfully processed {len(row_dicts)} records from Excel bytes"
            )
            return row_dicts

        except Exception as e:
            logger.error(f"Error processing Excel bytes for {filename}: {e}")
            raise

    def get_sheet_names(self, file_path: str) -> List[str]:
        """
        Get all sheet names from an Excel file.

        Args:
            file_path: Path to the Excel file

        Returns:
            List of sheet names
        """
        try:
            excel_file = pd.ExcelFile(file_path)
            sheet_names = excel_file.sheet_names
            logger.info(f"Found {len(sheet_names)} sheets: {sheet_names}")
            return sheet_names

        except Exception as e:
            logger.error(f"Error getting sheet names from {file_path}: {e}")
            return []

    def get_sheet_names_from_bytes(self, file_bytes: bytes) -> List[str]:
        """
        Get all sheet names from Excel bytes.

        Args:
            file_bytes: Excel file as bytes

        Returns:
            List of sheet names
        """
        try:
            excel_buffer = io.BytesIO(file_bytes)
            excel_file = pd.ExcelFile(excel_buffer)
            sheet_names = excel_file.sheet_names
            logger.info(f"Found {len(sheet_names)} sheets: {sheet_names}")
            return sheet_names

        except Exception as e:
            logger.error(f"Error getting sheet names from bytes: {e}")
            return []
