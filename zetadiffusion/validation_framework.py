"""
validation_framework.py

Shared validation framework for consistent result handling, execution tracking,
and Notion uploads. Eliminates code duplication across validation scripts.

Author: Joel
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, Callable
from functools import wraps
from dataclasses import dataclass, asdict
from datetime import datetime

# Import notion_logger - handle both package and root-level imports
try:
    from notion_logger import ValidationRunLogger
except ImportError:
    try:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from notion_logger import ValidationRunLogger
    except ImportError:
        # Notion upload disabled if logger unavailable
        ValidationRunLogger = None

@dataclass
class ValidationResult:
    """Standardized validation result structure."""
    validation_type: str
    parameters: Dict[str, Any]
    results: Dict[str, Any]
    execution_time: float
    timestamp: str
    success: bool
    output_files: list = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        # Ensure all values are JSON-serializable
        return json.loads(json.dumps(result, default=str))

class ValidationRunner:
    """
    Shared validation runner that handles:
    - Execution time tracking
    - Result serialization
    - File output
    - Notion uploads
    - Error handling
    """
    
    def __init__(self, output_dir: str = ".out", auto_upload: bool = True):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.auto_upload = auto_upload
        self.logger = ValidationRunLogger() if (auto_upload and ValidationRunLogger) else None
    
    def run(
        self,
        validation_type: str,
        validation_func: Callable,
        parameters: Optional[Dict] = None,
        output_filename: Optional[str] = None,
        **func_kwargs
    ) -> ValidationResult:
        """
        Run a validation function with full tracking and upload.
        
        Args:
            validation_type: Name of validation (e.g., "Operator Analysis")
            validation_func: Function that returns results dict
            parameters: Validation parameters
            output_filename: Optional custom filename (default: {type}_results.json)
            **func_kwargs: Arguments to pass to validation_func
        
        Returns:
            ValidationResult with all metadata
        """
        start_time = time.time()
        parameters = parameters or {}
        
        print(f"\n{'='*70}")
        print(f"Running: {validation_type}")
        print(f"{'='*70}")
        
        try:
            # Run validation
            results = validation_func(**func_kwargs)
            
            # Ensure results is a dict
            if not isinstance(results, dict):
                results = {"result": results}
            
            execution_time = time.time() - start_time
            
            # Create result object
            if output_filename is None:
                safe_name = validation_type.lower().replace(" ", "_").replace(".", "")
                output_filename = f"{safe_name}_results.json"
            
            output_file = self.output_dir / output_filename
            
            # Add metadata
            results['execution_time'] = execution_time
            results['timestamp'] = datetime.now().isoformat()
            results['success'] = results.get('success', True)
            results['validation_type'] = validation_type
            
            # Save to file
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            print(f"\n✓ Results saved to: {output_file}")
            print(f"✓ Execution time: {execution_time:.3f} seconds")
            
            # Create validation result
            validation_result = ValidationResult(
                validation_type=validation_type,
                parameters=parameters,
                results=results,
                execution_time=execution_time,
                timestamp=results['timestamp'],
                success=results['success'],
                output_files=[str(output_file)]
            )
            
            # Upload to Notion (includes reports and plots automatically)
            if self.auto_upload and self.logger:
                try:
                    page_id = self.logger.create_run_entry(
                        validation_type=validation_type,
                        parameters=parameters,
                        results=results,
                        execution_time=execution_time,
                        output_files=[str(output_file)]
                    )
                    if page_id:
                        print(f"✓ Uploaded to Notion (with reports & plots): {page_id}")
                except Exception as e:
                    print(f"⚠ Notion upload failed: {e}")
            
            return validation_result
            
        except Exception as e:
            execution_time = time.time() - start_time
            print(f"\n✗ Validation failed: {e}")
            
            # Save error result
            error_result = {
                'error': str(e),
                'execution_time': execution_time,
                'timestamp': datetime.now().isoformat(),
                'success': False
            }
            
            error_file = self.output_dir / f"{validation_type.lower().replace(' ', '_')}_error.json"
            with open(error_file, 'w') as f:
                json.dump(error_result, f, indent=2)
            
            return ValidationResult(
                validation_type=validation_type,
                parameters=parameters,
                results=error_result,
                execution_time=execution_time,
                timestamp=error_result['timestamp'],
                success=False,
                output_files=[str(error_file)]
            )
    
    def run_with_context(
        self,
        validation_type: str,
        parameters: Optional[Dict] = None,
        output_filename: Optional[str] = None
    ):
        """
        Decorator for validation functions.
        
        Usage:
            @runner.run_with_context("My Validation", parameters={'param': 'value'})
            def my_validation():
                # ... validation code ...
                return results_dict
        """
        def decorator(func: Callable):
            @wraps(func)
            def wrapper(*args, **kwargs):
                return self.run(
                    validation_type=validation_type,
                    validation_func=func,
                    parameters=parameters or {},
                    output_filename=output_filename,
                    **kwargs
                )
            return wrapper
        return decorator

# Global runner instance
_default_runner = ValidationRunner()

def run_validation(
    validation_type: str,
    validation_func: Callable,
    parameters: Optional[Dict] = None,
    output_filename: Optional[str] = None,
    **func_kwargs
) -> ValidationResult:
    """
    Convenience function using default runner.
    
    Usage:
        def my_validation():
            return {'result': 'data'}
        
        result = run_validation("My Validation", my_validation, parameters={'p': 1})
    """
    return _default_runner.run(
        validation_type=validation_type,
        validation_func=validation_func,
        parameters=parameters,
        output_filename=output_filename,
        **func_kwargs
    )

