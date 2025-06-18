"""
Streaming utilities for Anthropic API

This module provides utilities for handling streaming responses from the Anthropic API,
including event processing and response accumulation.
"""

import anthropic
from typing import Dict, Any, List, Optional, Iterator, AsyncIterator
from dataclasses import dataclass, field
from collections.abc import Mapping
from frozendict import frozendict
import json


@dataclass(frozen=True)
class StreamingConfig:
    """Immutable configuration for streaming operations."""
    max_tokens: int = 65536
    model: str = "claude-sonnet-4-20250514"
    betas: Optional[List[str]] = None
    tools: Optional[List[Dict[str, Any]]] = None


class StreamingResponse:
    """
    Wrapper for handling streaming responses from Anthropic API.
    """
    
    def __init__(self, stream_response):
        self.stream_response = stream_response
        self._accumulated_text = ""
        self._usage = None
    
    def __iter__(self) -> Iterator[str]:
        """Iterate over text chunks in the stream."""
        for text in self.stream_response.text_stream:
            self._accumulated_text += text
            yield text
    
    async def __aiter__(self) -> AsyncIterator[str]:
        """Async iterate over text chunks in the stream."""
        async for text in self.stream_response.text_stream:
            self._accumulated_text += text
            yield text
    
    @property
    def accumulated_text(self) -> str:
        """Get all accumulated text from the stream."""
        return self._accumulated_text
    
    @property
    def usage(self) -> Optional[Dict[str, Any]]:
        """Get usage information from the stream."""
        if hasattr(self.stream_response, 'usage'):
            return self.stream_response.usage
        return None


class StreamingTools:
    """
    Collection of utilities for working with streaming responses.
    """
    
    @staticmethod
    def create_streaming_client(client: anthropic.Anthropic, config: StreamingConfig) -> Any:
        """
        Create a streaming client with the given configuration.
        
        Args:
            client: The Anthropic client instance
            config: Streaming configuration
            
        Returns:
            Streaming client instance
        """
        return client.messages.stream(
            max_tokens=config.max_tokens,
            model=config.model,
            betas=config.betas,
            tools=config.tools
        )
    
    @staticmethod
    def process_streaming_events(stream_response) -> Iterator[Dict[str, Any]]:
        """
        Process streaming events and yield event data.
        
        Args:
            stream_response: The streaming response object
            
        Yields:
            Event data dictionaries
        """
        for event in stream_response:
            if hasattr(event, 'type'):
                yield {
                    'type': event.type,
                    'data': event.data if hasattr(event, 'data') else None,
                    'delta': event.delta if hasattr(event, 'delta') else None
                }
    
    @staticmethod
    def accumulate_streaming_text(stream_response) -> str:
        """
        Accumulate all text from a streaming response.
        
        Args:
            stream_response: The streaming response object
            
        Returns:
            Accumulated text string
        """
        accumulated = ""
        for text in stream_response.text_stream:
            accumulated += text
        return accumulated


def example_streaming_with_tools(client: anthropic.Anthropic) -> StreamingResponse:
    """
    Example of using streaming with tools.
    
    Args:
        client: The Anthropic client instance
        
    Returns:
        StreamingResponse object
    """
    from .tools import StreamingTools as Tools
    
    config = StreamingConfig(
        max_tokens=65536,
        model="claude-sonnet-4-20250514",
        betas=["fine-grained-tool-streaming-2025-05-14"],
        tools=[Tools.make_file_tool()]
    )
    
    messages = [{
        "role": "user",
        "content": "Can you write a long poem and make a file called poem.txt?"
    }]
    
    stream_response = client.messages.stream(
        max_tokens=config.max_tokens,
        model=config.model,
        tools=config.tools,
        messages=messages,
        betas=config.betas
    )
    
    return StreamingResponse(stream_response)


def example_basic_streaming(client: anthropic.Anthropic) -> StreamingResponse:
    """
    Example of basic streaming without tools.
    
    Args:
        client: The Anthropic client instance
        
    Returns:
        StreamingResponse object
    """
    config = StreamingConfig(
        max_tokens=1024,
        model="claude-sonnet-4-20250514"
    )
    
    messages = [{"role": "user", "content": "Write a short story about a robot learning to paint."}]
    
    stream_response = client.messages.stream(
        max_tokens=config.max_tokens,
        model=config.model,
        messages=messages
    )
    
    return StreamingResponse(stream_response)


# Utility functions for working with streaming data
def print_streaming_response(streaming_response: StreamingResponse) -> None:
    """
    Print a streaming response in real-time.
    
    Args:
        streaming_response: The streaming response object
    """
    print("🤖 Streaming Response:")
    for text_chunk in streaming_response:
        print(text_chunk, end="", flush=True)
    print()  # New line at the end


def save_streaming_response(streaming_response: StreamingResponse, filename: str) -> None:
    """
    Save a streaming response to a file.
    
    Args:
        streaming_response: The streaming response object
        filename: The filename to save to
    """
    with open(filename, 'w', encoding='utf-8') as f:
        for text_chunk in streaming_response:
            f.write(text_chunk)
    
    print(f"✅ Streaming response saved to {filename}")


def get_streaming_usage(streaming_response: StreamingResponse) -> Optional[Dict[str, Any]]:
    """
    Get usage information from a streaming response.
    
    Args:
        streaming_response: The streaming response object
        
    Returns:
        Usage information dictionary or None
    """
    return streaming_response.usage 