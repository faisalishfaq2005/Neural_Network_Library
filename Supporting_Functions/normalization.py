import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional, List, Union
import warnings

class DataTypeHandler:
    """Utility class to detect and handle different data types"""
    
    @staticmethod
    def is_numeric(data: np.ndarray) -> bool:
        """Check if data is numeric"""
        try:
            np.asarray(data, dtype=float)
            return True
        except (ValueError, TypeError):
            return False
    
    @staticmethod
    def is_categorical(column: np.ndarray) -> bool:
        """Check if a column contains categorical/string data"""
        try:
            # Check if values are strings
            if column.dtype == object or column.dtype.kind in ['U', 'S', 'O']:
                # Try to convert to float; if it fails, it's categorical
                try:
                    np.asarray(column, dtype=float)
                    return False
                except (ValueError, TypeError):
                    return True
            return False
        except Exception:
            return True
    
    @staticmethod
    def detect_column_types(data: np.ndarray) -> List[str]:
        """Detect type of each column in 2D array"""
        try:
            if len(data.shape) == 1:
                return ['categorical' if DataTypeHandler.is_categorical(data) else 'numeric']
            
            column_types = []
            for i in range(data.shape[1]):
                column = data[:, i]
                if DataTypeHandler.is_categorical(column):
                    column_types.append('categorical')
                else:
                    column_types.append('numeric')
            return column_types
        except Exception:
            # If shape access fails, return empty list
            return []


class CategoricalEncoder:
    """Handle categorical data encoding/decoding"""
    
    def __init__(self):
        self.encoders = {}
        self.categories = {}
        
    def fit_transform(self, column: np.ndarray, method: str = 'label') -> np.ndarray:
        """
        Fit and transform categorical column
        Methods: 'label' (0, 1, 2...) or 'onehot' (one-hot encoding)
        """
        try:
            if method == 'label':
                unique_values = np.unique(column)
                self.categories[id(column)] = unique_values
                mapping = {val: idx for idx, val in enumerate(unique_values)}
                return np.array([mapping[val] for val in column]).reshape(-1, 1)
            
            elif method == 'onehot':
                unique_values = np.unique(column)
                self.categories[id(column)] = unique_values
                n_categories = len(unique_values)
                encoded = np.zeros((len(column), n_categories))
                
                for idx, val in enumerate(unique_values):
                    mask = column == val
                    encoded[mask, idx] = 1
                
                return encoded
            else:
                raise ValueError(f"Unknown encoding method: {method}")
                
        except Exception as e:
            raise ValueError(f"Error encoding categorical column: {str(e)}")
    
    def transform(self, column: np.ndarray, method: str = 'label') -> np.ndarray:
        """Transform using fitted encoders"""
        try:
            if method == 'label':
                unique_values = list(self.categories.values())[0]
                mapping = {val: idx for idx, val in enumerate(unique_values)}
                return np.array([mapping.get(val, -1) for val in column]).reshape(-1, 1)
            
            elif method == 'onehot':
                unique_values = list(self.categories.values())[0]
                n_categories = len(unique_values)
                encoded = np.zeros((len(column), n_categories))
                
                for idx, val in enumerate(unique_values):
                    mask = column == val
                    encoded[mask, idx] = 1
                
                return encoded
        except Exception as e:
            raise ValueError(f"Error transforming categorical column: {str(e)}")


def normalize(data: np.ndarray, min_val: Optional[np.ndarray] = None, 
              max_val: Optional[np.ndarray] = None, 
              handle_categorical: bool = True,
              categorical_method: str = 'label') -> Tuple[np.ndarray, Dict[str, Any], np.ndarray]:
    """
    Intelligently normalizes mixed data (numeric and categorical).
    
    Args:
        data: Input data array (can contain numeric and/or categorical columns)
        min_val: Pre-calculated minimum values for numeric normalization
        max_val: Pre-calculated maximum values for numeric normalization
        handle_categorical: Whether to encode categorical columns (default: True)
        categorical_method: 'label' or 'onehot' encoding for categorical data
    
    Returns:
        Tuple of:
        - normalized_data: Processed numeric data
        - metadata: Dictionary containing transformation information
        - encoded_categorical: Encoded categorical columns (if present)
    
    Raises:
        ValueError: If data is invalid or processing fails
        TypeError: If data type is not supported
    """
    try:
        # Input validation
        if data is None:
            raise ValueError("Input data cannot be None or empty")
        
        # Ensure numpy array
        if not isinstance(data, np.ndarray):
            try:
                data = np.asarray(data)
            except Exception as e:
                raise TypeError(f"Cannot convert input to numpy array: {str(e)}")
        
        # Check if empty
        if data.size == 0:
            raise ValueError("Input data cannot be None or empty")
        
        # Handle 0D arrays (scalar arrays) and 1D arrays
        if len(data.shape) == 0:
            raise ValueError("Input must be at least 1D array")
        elif len(data.shape) == 1:
            data = data.reshape(-1, 1)
        
        metadata = {
            'original_shape': data.shape,
            'column_types': DataTypeHandler.detect_column_types(data),
            'numeric_indices': [],
            'categorical_indices': [],
            'min_values': None,
            'max_values': None,
            'categorical_encoders': {},
            'categorical_method': categorical_method
        }
        
        # Separate numeric and categorical columns
        numeric_columns = []
        categorical_columns = []
        encoded_categorical_list = []
        
        # Handle case where data becomes 1D after conversion
        if len(data.shape) == 1:
            # Single column case
            if metadata['column_types'] and metadata['column_types'][0] == 'numeric':
                metadata['numeric_indices'].append(0)
                try:
                    numeric_columns.append(np.asarray(data, dtype=float))
                except (ValueError, TypeError):
                    metadata['column_types'][0] = 'categorical'
                    categorical_columns.append(data)
                    metadata['categorical_indices'].append(0)
            else:
                metadata['categorical_indices'].append(0)
                categorical_columns.append(data)
        else:
            for i in range(data.shape[1]):
                column = data[:, i]
                
                if metadata['column_types'][i] == 'numeric':
                    metadata['numeric_indices'].append(i)
                    try:
                        numeric_columns.append(np.asarray(column, dtype=float))
                    except (ValueError, TypeError) as e:
                        warnings.warn(f"Column {i} could not be converted to float, treating as categorical: {str(e)}")
                        metadata['column_types'][i] = 'categorical'
                        categorical_columns.append(column)
                        metadata['categorical_indices'].append(i)
                        
                else:
                    metadata['categorical_indices'].append(i)
                    categorical_columns.append(column)
        
        # Process numeric columns
        if numeric_columns:
            numeric_data = np.column_stack(numeric_columns)
            
            try:
                # Check for NaN or Inf values
                if np.any(np.isnan(numeric_data)) or np.any(np.isinf(numeric_data)):
                    warnings.warn("Data contains NaN or Inf values. These will be replaced with column mean.")
                    numeric_data = handle_missing_values(numeric_data)
                
                # Normalize numeric data
                if min_val is None:
                    min_val = np.nanmin(numeric_data, axis=0)
                if max_val is None:
                    max_val = np.nanmax(numeric_data, axis=0)
                
                # Handle division by zero (constant columns)
                diff = max_val - min_val
                diff[diff == 0] = 1  # Avoid division by zero for constant columns
                
                normalized_numeric = (numeric_data - min_val) / diff
                metadata['min_values'] = min_val
                metadata['max_values'] = max_val
                
            except Exception as e:
                raise ValueError(f"Error normalizing numeric columns: {str(e)}")
        else:
            normalized_numeric = None
        
        # Process categorical columns
        if categorical_columns and handle_categorical:
            encoder = CategoricalEncoder()
            for col in categorical_columns:
                try:
                    encoded = encoder.fit_transform(col, method=categorical_method)
                    encoded_categorical_list.append(encoded)
                    metadata['categorical_encoders'][len(encoded_categorical_list)-1] = encoder.categories
                except Exception as e:
                    raise ValueError(f"Error encoding categorical column: {str(e)}")
        
        # Combine results
        if normalized_numeric is not None and encoded_categorical_list:
            # Interleave numeric and categorical columns in original order
            combined_data = combine_columns(
                normalized_numeric, 
                encoded_categorical_list, 
                metadata['numeric_indices'],
                metadata['categorical_indices']
            )
            encoded_categorical = np.hstack(encoded_categorical_list)
        elif normalized_numeric is not None:
            combined_data = normalized_numeric
            encoded_categorical = None
        elif encoded_categorical_list:
            combined_data = np.hstack(encoded_categorical_list)
            encoded_categorical = None
        else:
            raise ValueError("No valid data to process")
        
        return combined_data.astype(np.float32), metadata, encoded_categorical
        
    except (ValueError, TypeError) as e:
        raise
    except Exception as e:
        raise ValueError(f"Unexpected error during normalization: {str(e)}")


def denormalize(normalized_data: np.ndarray, metadata: Dict[str, Any]) -> np.ndarray:
    """
    Rescales normalized numeric data back to original range.
    
    Args:
        normalized_data: Normalized data
        metadata: Metadata dictionary from normalize()
    
    Returns:
        Denormalized data
    """
    try:
        if metadata['min_values'] is None or metadata['max_values'] is None:
            raise ValueError("Cannot denormalize without min/max values in metadata")
        
        min_val = metadata['min_values']
        max_val = metadata['max_values']
        diff = max_val - min_val
        diff[diff == 0] = 1
        
        return normalized_data * diff + min_val
        
    except Exception as e:
        raise ValueError(f"Error denormalizing data: {str(e)}")


def handle_missing_values(data: np.ndarray) -> np.ndarray:
    """Handle NaN and Inf values by replacing with column mean"""
    try:
        data = data.astype(float)
        for i in range(data.shape[1]):
            col = data[:, i]
            mask = np.isnan(col) | np.isinf(col)
            if np.any(mask):
                valid_values = col[~mask]
                if len(valid_values) > 0:
                    col_mean = np.nanmean(valid_values)
                    col[mask] = col_mean
        return data
    except Exception as e:
        raise ValueError(f"Error handling missing values: {str(e)}")


def combine_columns(numeric_data: np.ndarray, 
                   categorical_data: List[np.ndarray],
                   numeric_indices: List[int],
                   categorical_indices: List[int]) -> np.ndarray:
    """Combine numeric and categorical columns in original order"""
    try:
        total_cols = len(numeric_indices) + len(categorical_indices)
        result_cols = []
        numeric_idx = 0
        categorical_idx = 0
        
        for i in range(total_cols):
            if i in numeric_indices:
                result_cols.append(numeric_data[:, numeric_idx])
                numeric_idx += 1
            else:
                result_cols.append(categorical_data[categorical_idx])
                categorical_idx += 1
        
        # Flatten categorical data if needed and stack
        final_cols = []
        for col in result_cols:
            if len(col.shape) == 1:
                final_cols.append(col.reshape(-1, 1))
            else:
                final_cols.append(col)
        
        return np.hstack(final_cols)
    except Exception as e:
        raise ValueError(f"Error combining columns: {str(e)}")


def preprocess_dataset(data: Union[np.ndarray, pd.DataFrame],
                      handle_categorical: bool = True,
                      categorical_method: str = 'label',
                      remove_duplicates: bool = False) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Complete preprocessing pipeline for datasets
    
    Args:
        data: Input dataset
        handle_categorical: Whether to encode categorical columns
        categorical_method: 'label' or 'onehot' encoding
        remove_duplicates: Whether to remove duplicate rows
    
    Returns:
        Tuple of (processed_data, metadata)
    """
    try:
        # Convert pandas DataFrame to numpy array if needed
        if isinstance(data, pd.DataFrame):
            data = data.values
        
        # Remove duplicates if requested
        if remove_duplicates:
            data = np.unique(data, axis=0)
        
        # Normalize data
        processed_data, metadata, _ = normalize(
            data,
            handle_categorical=handle_categorical,
            categorical_method=categorical_method
        )
        
        return processed_data, metadata
        
    except Exception as e:
        raise ValueError(f"Error in dataset preprocessing: {str(e)}")
