# -*- coding: utf-8 -*-
"""
Prompt template implementation
"""

from typing import Any, Dict, List, Optional

from my_langchain.prompts.base import BasePromptTemplate
from my_langchain.prompts.types import (
    PromptTemplateConfig, PromptTemplateResult, PromptInputVariables,
    TemplateValidationError, TemplateFormatError
)
from pydantic import Field


class PromptTemplate(BasePromptTemplate):
    """
    Simple prompt template implementation using f-string formatting
    """

    partial_variables: Dict[str, Any] = Field(default_factory=dict, description="Partial variables")

    def __init__(
        self,
        template: str,
        input_variables: Optional[List[str]] = None,
        partial_variables: Optional[Dict[str, Any]] = None,
        template_format: str = "f-string",
        validate_template: bool = True,
        **kwargs
    ):
        """
        Initialize prompt template

        Args:
            template: Template string with variables in {variable} format
            input_variables: List of expected input variables (auto-extracted if None)
            partial_variables: Partial variables to always include
            template_format: Template format (currently only f-string supported)
            validate_template: Whether to validate template
            **kwargs: Additional parameters
        """
        # Extract input variables if not provided
        if input_variables is None:
            input_variables = self._extract_variables_from_template(template)

        # Create configuration
        config = PromptTemplateConfig(
            template_format=template_format,
            validate_template=validate_template
        )

        # Initialize partial variables
        partial_vars = partial_variables or {}

        super().__init__(
            template=template,
            input_variables=input_variables,
            config=config,
            partial_variables=partial_vars,
            **kwargs
        )

    def format(self, **kwargs) -> str:
        """
        Format the template with provided variables

        Args:
            **kwargs: Variables to substitute in template

        Returns:
            Formatted template string
        """
        result = self.format_with_result(**kwargs)
        return result.text

    def format_with_result(self, **kwargs) -> PromptTemplateResult:
        """
        Format template and return detailed result

        Args:
            **kwargs: Variables to substitute in template

        Returns:
            Formatted result with metadata
        """
        # Merge partial variables with provided variables
        merged_vars = self._merge_partial_variables(kwargs, self.partial_variables)

        # Validate input variables
        if self.config.strict_variables:
            self._validate_input_variables(merged_vars)

        # Format template
        if self.config.template_format == "f-string":
            formatted_text = self._format_f_string(self.template, merged_vars)
        else:
            raise TemplateFormatError(
                f"Unsupported template format: {self.config.template_format}",
                self.template,
                self.config.template_format
            )

        # Find missing variables
        missing_vars = []
        if self.config.strict_variables:
            for var in self.input_variables:
                if var not in merged_vars:
                    missing_vars.append(var)

        return PromptTemplateResult(
            text=formatted_text,
            variables=merged_vars,
            missing_variables=missing_vars,
            template_name=getattr(self, 'name', None),
            metadata={
                "template_format": self.config.template_format,
                "partial_variables": list(self.partial_variables.keys())
            }
        )

    def _extract_variables(self) -> List[str]:
        """
        Extract variable names from template

        Returns:
            List of variable names found in template
        """
        return self._extract_variables_from_template(self.template)

    def _extract_variables_from_template(self, template: str) -> List[str]:
        """
        Extract variables from template string

        Args:
            template: Template string

        Returns:
            List of variable names
        """
        # Default to f-string format if config is not available yet
        template_format = getattr(self.config, 'template_format', 'f-string') if hasattr(self, 'config') else 'f-string'

        if template_format == "f-string":
            return self._extract_f_string_variables(template)
        else:
            raise TemplateFormatError(
                f"Unsupported template format for variable extraction: {template_format}",
                template,
                template_format
            )

    def _validate_template(self) -> None:
        """
        Validate template format and variables

        Raises:
            TemplateValidationError: If template is invalid
        """
        try:
            # Extract variables from template
            extracted_vars = self._extract_variables()

            # Check if input_variables match extracted variables
            if set(self.input_variables) != set(extracted_vars):
                if not self.input_variables:  # If input_variables is empty, use extracted
                    self.input_variables = extracted_vars
                else:
                    raise TemplateValidationError(
                        f"Input variables {self.input_variables} do not match template variables {extracted_vars}",
                        self.template,
                        extracted_vars
                    )

            # Check template syntax by creating a dummy format string
            template_format = getattr(self.config, 'template_format', 'f-string') if hasattr(self, 'config') else 'f-string'
            if template_format == "f-string":
                # Create test variables for validation with appropriate types
                test_vars = {}
                for var in self.input_variables:
                    # Use appropriate test values based on common variable names
                    if var.lower() in ['age', 'count', 'number', 'score']:
                        test_vars[var] = 1  # Use integer for numeric variables
                    else:
                        test_vars[var] = f"test_{var}"  # Use string for text variables

                # This will raise ValueError if template syntax is invalid
                self.template.format(**test_vars)

        except (ValueError, KeyError) as e:
            raise TemplateValidationError(
                f"Invalid template syntax: {str(e)}",
                self.template,
                self.input_variables
            )

    def partial(self, **kwargs) -> 'PromptTemplate':
        """
        Create a new template with partial variables

        Args:
            **kwargs: Partial variables to set

        Returns:
            New template instance with partial variables
        """
        new_partial = {**self.partial_variables, **kwargs}
        return PromptTemplate(
            template=self.template,
            input_variables=self.input_variables,
            partial_variables=new_partial,
            template_format=self.config.template_format,
            validate_template=self.config.validate_template
        )

    def invoke(self, input: Any) -> str:
        """
        Invoke template with input

        Args:
            input: Input data (can be dict or any object)

        Returns:
            Formatted template string
        """
        if isinstance(input, dict):
            return self.format(**input)
        else:
            # For single input, use first variable
            if self.input_variables:
                return self.format(**{self.input_variables[0]: input})
            else:
                return self.format()

    def save(self, file_path: str) -> None:
        """
        Save template to file

        Args:
            file_path: Path to save template
        """
        self.save_template(file_path)

    @classmethod
    def load(cls, file_path: str) -> 'PromptTemplate':
        """
        Load template from file

        Args:
            file_path: Path to template file

        Returns:
            Loaded template instance
        """
        import json

        with open(file_path, 'r', encoding='utf-8') as f:
            template_data = json.load(f)

        return cls(
            template=template_data["template"],
            input_variables=template_data["input_variables"],
            partial_variables=template_data.get("partial_variables", {}),
            **template_data.get("config", {})
        )