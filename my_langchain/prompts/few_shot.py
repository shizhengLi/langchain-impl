# -*- coding: utf-8 -*-
"""
Few-shot prompt template implementation
"""

from typing import Any, Dict, List, Optional, Tuple

from my_langchain.prompts.prompt_template import PromptTemplate
from my_langchain.prompts.types import PromptTemplateResult
from pydantic import Field


class FewShotPromptTemplate(PromptTemplate):
    """
    Template for few-shot learning with examples
    """

    examples: List[Dict[str, Any]] = Field(default_factory=list, description="Few-shot examples")
    example_prompt: PromptTemplate = Field(..., description="Template for formatting examples")
    example_separator: str = Field(default="\n\n", description="Separator between examples")
    prefix: str = Field(default="", description="Text before examples")
    suffix: str = Field(default="", description="Text after examples")

    def __init__(
        self,
        examples: List[Dict[str, Any]],
        example_prompt: PromptTemplate,
        prefix: str = "",
        suffix: str = "",
        example_separator: str = "\n\n",
        **kwargs
    ):
        """
        Initialize few-shot template

        Args:
            examples: List of example dictionaries
            example_prompt: Template for formatting each example
            prefix: Text to add before examples
            suffix: Text to add after examples
            example_separator: Separator between examples
            **kwargs: Additional parameters
        """
        # Build the full template
        template = self._build_template(prefix, suffix, example_separator)

        # Extract input variables from suffix only (example_prompt variables are internal)
        input_variables = []

        # Add variables from suffix
        if suffix:
            suffix_vars = self._extract_f_string_variables(suffix)
            input_variables.extend(suffix_vars)

        # Remove duplicates
        input_variables = list(set(input_variables))

        super().__init__(
            template=template,
            input_variables=input_variables,
            examples=examples,
            example_prompt=example_prompt,
            prefix=prefix,
            suffix=suffix,
            example_separator=example_separator,
            **kwargs
        )

    def _build_template(self, prefix: str, suffix: str, separator: str) -> str:
        """
        Build the complete template string

        Args:
            prefix: Prefix text
            suffix: Suffix text
            separator: Example separator

        Returns:
            Complete template string
        """
        parts = []
        if prefix:
            parts.append(prefix)

        # Add placeholder for examples
        parts.append("{examples}")

        if suffix:
            parts.append(suffix)

        return separator.join(parts)

    def format(self, **kwargs) -> str:
        """
        Format the few-shot template

        Args:
            **kwargs: Variables for the suffix

        Returns:
            Formatted template string
        """
        result = self.format_with_result(**kwargs)
        return result.text

    def format_with_result(self, **kwargs) -> PromptTemplateResult:
        """
        Format few-shot template and return detailed result

        Args:
            **kwargs: Variables for the suffix

        Returns:
            Formatted result with metadata
        """
        # Format examples
        formatted_examples = []
        for example in self.examples:
            formatted_example = self.example_prompt.format(**example)
            formatted_examples.append(formatted_example)

        examples_text = self.example_separator.join(formatted_examples)

        # Build final variables
        template_vars = {"examples": examples_text, **kwargs}

        # Format the main template
        return super().format_with_result(**template_vars)

    def add_example(self, example: Dict[str, Any]) -> None:
        """
        Add an example to the template

        Args:
            example: Example dictionary
        """
        self.examples.append(example)

    def add_examples(self, examples: List[Dict[str, Any]]) -> None:
        """
        Add multiple examples to the template

        Args:
            examples: List of example dictionaries
        """
        self.examples.extend(examples)

    def clear_examples(self) -> None:
        """Clear all examples"""
        self.examples = []

    def get_example_count(self) -> int:
        """
        Get number of examples

        Returns:
            Number of examples
        """
        return len(self.examples)

    def _validate_template(self) -> None:
        """
        Validate few-shot template structure

        For FewShotTemplate, we only validate that the template contains the expected
        placeholders: {examples} and variables from suffix
        """
        try:
            # Check that the template contains {examples} placeholder
            if "{examples}" not in self.template:
                raise ValueError("FewShot template must contain {examples} placeholder")

            # Extract variables from the template (should include 'examples' and suffix variables)
            extracted_vars = self._extract_f_string_variables(self.template)

            # For few-shot, expected variables are: examples + suffix variables
            expected_vars = set(self.input_variables)
            expected_vars.add("examples")  # Add the examples placeholder

            # Check if variables match
            if set(extracted_vars) != expected_vars:
                # It's ok if there are extra variables in input_variables from example_prompt
                # They will be used in example formatting
                pass

            # Validate template syntax by creating test variables
            test_vars = {}
            for var in extracted_vars:
                if var == "examples":
                    test_vars[var] = "Example content"
                else:
                    test_vars[var] = f"test_{var}"

            self.template.format(**test_vars)

        except (ValueError, KeyError) as e:
            from my_langchain.prompts.types import TemplateValidationError
            raise TemplateValidationError(
                f"Invalid few-shot template: {str(e)}",
                self.template,
                self.input_variables
            )

    def select_examples(self, input_variables: Dict[str, Any], max_examples: int) -> List[Dict[str, Any]]:
        """
        Select examples based on input (simple implementation)

        Args:
            input_variables: Input variables
            max_examples: Maximum number of examples to return

        Returns:
            Selected examples
        """
        # Simple implementation: return first N examples
        # Subclasses can override with more sophisticated selection
        return self.examples[:max_examples]

    def save(self, file_path: str) -> None:
        """
        Save few-shot template to file

        Args:
            file_path: Path to save template
        """
        import json

        template_data = {
            "template": self.template,
            "input_variables": self.input_variables,
            "examples": self.examples,
            "example_prompt": self.example_prompt.get_template_info(),
            "prefix": self.prefix,
            "suffix": self.suffix,
            "example_separator": self.example_separator,
            "config": self.config.model_dump(),
            "type": self.__class__.__name__
        }

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(template_data, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, file_path: str) -> 'FewShotPromptTemplate':
        """
        Load few-shot template from file

        Args:
            file_path: Path to template file

        Returns:
            Loaded template instance
        """
        import json

        with open(file_path, 'r', encoding='utf-8') as f:
            template_data = json.load(f)

        # Recreate example prompt
        from my_langchain.prompts.prompt_template import PromptTemplate
        example_prompt_info = template_data["example_prompt"]
        example_prompt = PromptTemplate(
            template=example_prompt_info["template"],
            input_variables=example_prompt_info["input_variables"]
        )

        return cls(
            examples=template_data["examples"],
            example_prompt=example_prompt,
            prefix=template_data.get("prefix", ""),
            suffix=template_data.get("suffix", ""),
            example_separator=template_data.get("example_separator", "\n\n")
        )