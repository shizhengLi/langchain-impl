# -*- coding: utf-8 -*-
"""
Chat prompt template implementation
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Union

from my_langchain.prompts.base import BasePromptTemplate
from my_langchain.prompts.types import PromptTemplateConfig, PromptTemplateResult, VariableMissingError
from pydantic import BaseModel, Field


class ChatMessageType(str, Enum):
    """Chat message types"""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"


class ChatMessage(BaseModel):
    """Chat message model"""
    role: ChatMessageType = Field(..., description="Message role")
    content: str = Field(..., description="Message content")
    name: Optional[str] = Field(default=None, description="Optional message name")


class ChatPromptTemplate(BasePromptTemplate):
    """
    Template for chat conversations with multiple roles
    """

    messages: List[ChatMessage] = Field(default_factory=list, description="List of chat messages")
    message_templates: List[Dict[str, Any]] = Field(default_factory=list, description="Message templates")

    def __init__(
        self,
        messages: Optional[List[Union[ChatMessage, Dict[str, Any]]]] = None,
        message_templates: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ):
        """
        Initialize chat prompt template

        Args:
            messages: List of messages or message dictionaries
            message_templates: List of message templates with variables
            **kwargs: Additional parameters
        """
        # Process messages
        processed_messages = []
        if messages:
            for msg in messages:
                if isinstance(msg, ChatMessage):
                    processed_messages.append(msg)
                else:
                    processed_messages.append(ChatMessage(**msg))

        # Build input variables from message templates
        input_variables = self._extract_chat_variables(message_templates or [])

        # Build template string for base class
        template = self._build_chat_template(message_templates or [])

        super().__init__(
            template=template,
            input_variables=input_variables,
            messages=processed_messages,
            message_templates=message_templates or [],
            **kwargs
        )

    def _build_chat_template(self, message_templates: List[Dict[str, Any]]) -> str:
        """
        Build template string for chat messages

        Args:
            message_templates: List of message templates

        Returns:
            Template string
        """
        template_parts = []
        for msg_template in message_templates:
            role = msg_template.get("role", "user")
            content = msg_template.get("content", "")
            template_parts.append(f"{role}: {content}")

        return "\n".join(template_parts)

    def _extract_chat_variables(self, message_templates: List[Dict[str, Any]]) -> List[str]:
        """
        Extract variables from chat message templates

        Args:
            message_templates: List of message templates

        Returns:
            List of variable names
        """
        variables = set()
        for msg_template in message_templates:
            content = msg_template.get("content", "")
            if content:
                # Extract f-string variables
                content_vars = self._extract_f_string_variables(content)
                variables.update(content_vars)

        return list(variables)

    def format(self, **kwargs) -> str:
        """
        Format chat messages into a single string

        Args:
            **kwargs: Variables to substitute

        Returns:
            Formatted chat conversation string
        """
        result = self.format_with_result(**kwargs)
        return result.text

    def format_with_result(self, **kwargs) -> PromptTemplateResult:
        """
        Format chat messages and return detailed result

        Args:
            **kwargs: Variables to substitute

        Returns:
            Formatted result with metadata
        """
        formatted_messages = []

        # Add static messages
        for msg in self.messages:
            formatted_messages.append(f"{msg.role.value}: {msg.content}")

        # Format template messages
        for msg_template in self.message_templates:
            role = msg_template.get("role", "user")
            content_template = msg_template.get("content", "")

            # Format content with variables
            try:
                formatted_content = self._format_f_string(content_template, kwargs)
            except (KeyError, VariableMissingError) as e:
                # If variable is missing, keep the template as-is
                formatted_content = content_template

            formatted_messages.append(f"{role}: {formatted_content}")

        # Join all messages
        formatted_text = "\n".join(formatted_messages)

        return PromptTemplateResult(
            text=formatted_text,
            variables=kwargs,
            missing_variables=[],  # Chat templates are more lenient
            metadata={
                "template_format": "chat",
                "message_count": len(formatted_messages)
            }
        )

    def _extract_variables(self) -> List[str]:
        """
        Extract variable names from chat templates

        Returns:
            List of variable names
        """
        return self._extract_chat_variables(self.message_templates)

    def _validate_template(self) -> None:
        """
        Validate chat template structure

        Raises:
            TemplateValidationError: If template is invalid
        """
        # Validate message templates
        for msg_template in self.message_templates:
            role = msg_template.get("role")
            if role and role not in [r.value for r in ChatMessageType]:
                raise ValueError(f"Invalid message role: {role}")

            content = msg_template.get("content", "")
            if content:
                # Extract variables from content and validate syntax
                try:
                    # Extract variables from this content
                    content_vars = self._extract_f_string_variables(content)
                    # Create test variables for validation
                    test_vars = {var: f"test_{var}" for var in content_vars}
                    # This will check if the template syntax is valid
                    content.format(**test_vars)
                except ValueError as e:
                    if "invalid format string" in str(e):
                        raise ValueError(f"Invalid content template: {content}")
                except KeyError:
                    # Expected error for missing variables, ignore in validation
                    pass

    def format_messages(self, **kwargs) -> List[ChatMessage]:
        """
        Format chat messages into structured message objects

        Args:
            **kwargs: Variables to substitute

        Returns:
            List of formatted chat messages
        """
        messages = []

        # Add static messages
        messages.extend(self.messages)

        # Format template messages
        for msg_template in self.message_templates:
            role = ChatMessageType(msg_template.get("role", "user"))
            content_template = msg_template.get("content", "")

            # Format content with variables
            try:
                formatted_content = self._format_f_string(content_template, kwargs)
            except KeyError:
                # If variable is missing, keep the template as-is
                formatted_content = content_template

            messages.append(ChatMessage(role=role, content=formatted_content))

        return messages

    def add_message(self, role: Union[ChatMessageType, str], content: str, name: Optional[str] = None) -> None:
        """
        Add a message to the template

        Args:
            role: Message role
            content: Message content
            name: Optional message name
        """
        if isinstance(role, str):
            role = ChatMessageType(role)

        self.messages.append(ChatMessage(role=role, content=content, name=name))

    def add_message_template(self, role: Union[ChatMessageType, str], content: str) -> None:
        """
        Add a message template to the template

        Args:
            role: Message role
            content: Message content template
        """
        if isinstance(role, str):
            role = ChatMessageType(role)

        self.message_templates.append({"role": role.value, "content": content})

    def system_message(self, content: str) -> None:
        """Add system message"""
        self.add_message(ChatMessageType.SYSTEM, content)

    def user_message(self, content: str) -> None:
        """Add user message"""
        # Check if content contains variables
        if '{' in content and '}' in content:
            self.add_message_template(ChatMessageType.USER, content)
        else:
            self.add_message(ChatMessageType.USER, content)

    def assistant_message(self, content: str) -> None:
        """Add assistant message"""
        # Check if content contains variables
        if '{' in content and '}' in content:
            self.add_message_template(ChatMessageType.ASSISTANT, content)
        else:
            self.add_message(ChatMessageType.ASSISTANT, content)

    @classmethod
    def from_messages(cls, messages: List[Union[ChatMessage, Dict[str, Any]]]) -> 'ChatPromptTemplate':
        """
        Create chat template from list of messages

        Args:
            messages: List of messages

        Returns:
            ChatPromptTemplate instance
        """
        return cls(messages=messages)

    def save(self, file_path: str) -> None:
        """
        Save chat template to file

        Args:
            file_path: Path to save template
        """
        import json

        # Convert messages to dict for serialization
        messages_data = []
        for msg in self.messages:
            messages_data.append({
                "role": msg.role.value,
                "content": msg.content,
                "name": msg.name
            })

        template_data = {
            "template": self.template,
            "input_variables": self.input_variables,
            "messages": messages_data,
            "message_templates": self.message_templates,
            "config": self.config.model_dump(),
            "type": self.__class__.__name__
        }

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(template_data, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, file_path: str) -> 'ChatPromptTemplate':
        """
        Load chat template from file

        Args:
            file_path: Path to template file

        Returns:
            Loaded template instance
        """
        import json

        with open(file_path, 'r', encoding='utf-8') as f:
            template_data = json.load(f)

        # Recreate messages
        messages = []
        for msg_data in template_data["messages"]:
            messages.append(ChatMessage(
                role=ChatMessageType(msg_data["role"]),
                content=msg_data["content"],
                name=msg_data.get("name")
            ))

        return cls(
            messages=messages,
            message_templates=template_data["message_templates"]
        )