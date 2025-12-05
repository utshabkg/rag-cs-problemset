"""
Excel reader module for CS-PROB.
Handles reading Excel workbooks and extracting embedded images.
"""
import base64
from pathlib import Path
from typing import Dict, Optional, List, Tuple

import pandas as pd
from openpyxl import load_workbook
from openpyxl.worksheet.worksheet import Worksheet

from config import IMAGE_BASE64_COLUMN


class ExcelReader:
    """Reads Excel workbooks and extracts questions with embedded images."""
    
    def __init__(self, file_path: Path, question_column_name: str):
        """
        Initialize Excel reader.
        
        Args:
            file_path: Path to Excel workbook
            question_column_name: Name of column containing questions
        """
        self.file_path = file_path
        self.question_column_name = question_column_name
        
    def read_workbook(self) -> Dict[str, pd.DataFrame]:
        """
        Read all sheets from workbook with embedded images.
        
        Returns:
            Dictionary mapping sheet names to DataFrames with ImageBase64 column
        """
        print(f"\n[INFO] Reading workbook: {self.file_path}")
        
        # Load data with pandas
        sheets_data = pd.read_excel(
            self.file_path, 
            sheet_name=None, 
            engine='openpyxl'
        )
        print(f"[INFO] Found {len(sheets_data)} sheet(s): {list(sheets_data.keys())}")
        
        # Load workbook with openpyxl for image extraction
        workbook = load_workbook(filename=str(self.file_path), data_only=True)
        
        result = {}
        for sheet_name, dataframe in sheets_data.items():
            print(f"\n[INFO] Processing sheet '{sheet_name}' ({len(dataframe)} rows)")
            
            worksheet = workbook[sheet_name]
            image_map = self._extract_images_from_worksheet(worksheet, sheet_name)
            dataframe_with_images = self._add_images_to_dataframe(
                dataframe, 
                image_map, 
                sheet_name
            )
            result[sheet_name] = dataframe_with_images
        
        return result
    
    def _extract_images_from_worksheet(
        self, 
        worksheet: Worksheet, 
        sheet_name: str
    ) -> Dict[Tuple[int, int], str]:
        """
        Extract embedded images from worksheet.
        
        Args:
            worksheet: openpyxl worksheet object
            sheet_name: Name of sheet (for logging)
            
        Returns:
            Dictionary mapping (row, col) to base64 encoded image
        """
        image_map = {}
        image_count = 0
        
        try:
            for image in worksheet._images:  # type: ignore[attr-defined]
                anchor = getattr(image, 'anchor', None)
                if not anchor:
                    continue
                
                # Handle different anchor types
                row_index, col_index = self._get_anchor_position(anchor)
                if row_index is None or col_index is None:
                    continue
                
                # Extract and encode image data
                image_bytes = image._data() if hasattr(image, '_data') else None
                if image_bytes:
                    base64_string = base64.b64encode(image_bytes).decode('utf-8')
                    image_map[(row_index, col_index)] = base64_string
                    image_count += 1
                    
        except Exception as error:
            print(f"[WARN] Could not extract images from '{sheet_name}': {error}")
        
        print(f"[INFO] Found {image_count} embedded image(s) in '{sheet_name}'")
        return image_map
    
    def _get_anchor_position(self, anchor) -> Tuple[Optional[int], Optional[int]]:
        """
        Get row and column from anchor object.
        
        Args:
            anchor: openpyxl anchor object
            
        Returns:
            Tuple of (row_index, col_index), 1-based indexing
        """
        # TwoCellAnchor type
        if hasattr(anchor, '_from'):
            return anchor._from.row + 1, anchor._from.col + 1
        
        # OneCellAnchor type
        if hasattr(anchor, 'row') and hasattr(anchor, 'col'):
            return anchor.row + 1, anchor.col + 1
        
        return None, None
    
    def _add_images_to_dataframe(
        self, 
        dataframe: pd.DataFrame, 
        image_map: Dict[Tuple[int, int], str],
        sheet_name: str
    ) -> pd.DataFrame:
        """
        Add ImageBase64 column to dataframe based on image map.
        
        Args:
            dataframe: Original DataFrame
            image_map: Dictionary mapping (row, col) to base64 image
            sheet_name: Sheet name for logging
            
        Returns:
            DataFrame with added ImageBase64 column
        """
        # Find question column index
        if self.question_column_name not in dataframe.columns:
            print(f"[WARN] Column '{self.question_column_name}' not found in '{sheet_name}'")
            return dataframe
        
        question_col_index = dataframe.columns.get_loc(self.question_column_name) + 1
        
        # Map images to rows
        image_base64_list: List[Optional[str]] = []
        for row_number in range(len(dataframe)):
            excel_row = row_number + 2  # Header is row 1, data starts at row 2
            base64_image = image_map.get((excel_row, question_col_index))
            image_base64_list.append(base64_image)
        
        # Add column
        dataframe = dataframe.copy()
        dataframe[IMAGE_BASE64_COLUMN] = image_base64_list
        
        embedded_count = sum(1 for img in image_base64_list if img is not None)
        print(f"[INFO] Mapped {embedded_count} image(s) to question rows")
        
        return dataframe


def read_image_from_file(image_path: Optional[str]) -> Optional[str]:
    """
    Read image file and convert to base64.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Base64 encoded image string, or None if file not found
    """
    if not image_path:
        return None
    
    file_path = Path(str(image_path)).expanduser()
    if not file_path.exists() or not file_path.is_file():
        return None
    
    with open(file_path, "rb") as file:
        return base64.b64encode(file.read()).decode("utf-8")
