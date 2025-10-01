# -*- coding: utf-8 -*-
"""
Base prompt template implementation
"""

import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

from my_langchain.base.base import BasePromptTemplate as BasePromptTemplateComponent
from my_langchain.prompts.types import (
    PromptTemplateConfig, PromptTemplateResult, PromptInputVariables,
    PromptTemplateError, TemplateValidationError, VariableMissingError
)
from pydantic import ConfigDict, Field


class BasePromptTemplate(BasePromptTemplateComponent):
    """
    Base implementation for prompt templates

    Provides common functionality for template validation, variable extraction,
    and formatting across different template formats.
    """

    config: PromptTemplateConfig = Field(..., description="Prompt template configuration")

    def __init__(
        self,
        template: str,
        input_variables: List[str],
        config: Optional[PromptTemplateConfig] = None,
        **kwargs
    ):
        """
        Initialize prompt template

        Args:
            template: Template string
            input_variables: List of expected input variables
            config: Template configuration
            **kwargs: Additional parameters
        """
        if config is None:
            config = PromptTemplateConfig()

        super().__init__(
            template=template,
            input_variables=input_variables,
            config=config,
            **kwargs
        )

        # Validate template if enabled
        if self.config.validate_template:
            self._validate_template()

    @abstractmethod
    def format(self, **kwargs) -> str:
        """
        Format the template with provided variables

        Args:
            **kwargs: Variables to substitute in template

        Returns:
            Formatted template string
        """
        pass

    @abstractmethod
    def format_with_result(self, **kwargs) -> PromptTemplateResult:
        """
        Format template and return detailed result

        Args:
            **kwargs: Variables to substitute in template

        Returns:
            Formatted result with metadata
        """
        pass

    @abstractmethod
    def _extract_variables(self) -> List[str]:
        """
        Extract variable names from template

        Returns:
            List of variable names found in template
        """
        pass

    @abstractmethod
    def _validate_template(self) -> None:
        """
        Validate template format and variables

        Raises:
            TemplateValidationError: If template is invalid
        """
        pass

    def _validate_input_variables(self, provided_vars: Dict[str, Any]) -> None:
        """
        Validate provided input variables

        Args:
            provided_vars: Dictionary of provided variables

        Raises:
            VariableMissingError: If required variables are missing
        """
        missing_vars = []
        for var in self.input_variables:
            if var not in provided_vars:
                missing_vars.append(var)

        if missing_vars:
            raise VariableMissingError(
                f"Missing required variables: {', '.join(missing_vars)}",
                missing_vars
            )

    def _merge_partial_variables(self, provided_vars: Dict[str, Any],
                                partial_vars: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Merge provided variables with partial variables

        Args:
            provided_vars: Variables provided at formatting time
            partial_vars: Partial variables set during initialization

        Returns:
            Merged variables dictionary
        """
        merged = {}
        if partial_vars:
            merged.update(partial_vars)
        merged.update(provided_vars)
        return merged

    def _extract_f_string_variables(self, template: str) -> List[str]:
        """
        Extract variables from f-string template

        Args:
            template: Template string

        Returns:
            List of variable names
        """
        # Match {variable} patterns, excluding escaped braces and numeric literals
        pattern = r'\{([^{}]+)\}'
        matches = re.findall(pattern, template)

        # Filter out format specifiers and complex expressions
        variables = []
        for match in matches:
            # Remove format specifiers (e.g., variable:format)
            var_name = match.split(':')[0].split('!')[0].strip()
            # Skip if it's not a simple variable name or is numeric
            if var_name.isidentifier() and not var_name.isdigit():
                variables.append(var_name)

        return list(set(variables))  # Remove duplicates

    def _format_f_string(self, template: str, variables: Dict[str, Any]) -> str:
        """
        Format f-string template with variables

        Args:
            template: Template string
            variables: Variables to substitute

        Returns:
            Formatted string
        """
        try:
            return template.format(**variables)
        except KeyError as e:
            missing_var = str(e).strip("'\"")
            raise VariableMissingError(
                f"Missing variable: {missing_var}",
                [missing_var]
            )
        except Exception as e:
            raise PromptTemplateError(f"Template formatting failed: {str(e)}")

    def get_template_info(self) -> Dict[str, Any]:
        """
        Get template information

        Returns:
            Template information dictionary
        """
        return {
            "template": self.template,
            "input_variables": self.input_variables,
            "template_format": self.config.template_format,
            "type": self.__class__.__name__
        }

    def save_template(self, file_path: str) -> None:
        """
        Save template to file

        Args:
            file_path: Path to save template
        """
        import json
        template_data = {
            "template": self.template,
            "input_variables": self.input_variables,
            "config": self.config.model_dump(),
            "type": self.__class__.__name__
        }

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(template_data, f, indent=2, ensure_ascii=False)

    @classmethod
    def load_template(cls, file_path: str) -> 'BasePromptTemplate':
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

        # This is a basic implementation - subclasses can override for custom loading
        return cls(
            template=template_data["template"],
            input_variables=template_data["input_variables"],
            **template_data.get("config", {})
        )