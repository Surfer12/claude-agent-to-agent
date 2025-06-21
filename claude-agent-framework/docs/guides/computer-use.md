# Computer Use Tool Implementation

## Overview

We have successfully implemented the computer use tool for the Claude Agent-to-Agent CLI. This tool enables Claude to interact with desktop environments through screenshots, mouse control, and keyboard input.

## Implementation Details

### Files Created/Modified

1. **`agents/tools/computer_use.py`** - Main computer use tool implementation
2. **`cli.py`** - Updated to include computer use tool options
3. **`agents/agent.py`** - Modified to support beta headers for computer use
4. **`README.md`** - Updated with computer use documentation
5. **`requirements.txt`** - Added MCP dependency
6. **`examples/computer_use_example.py`** - Example usage script

### Key Features

#### Tool Versions Supported
- `computer_20250124` (default) - Latest version with enhanced actions
- `computer_20241022` - Compatible with Claude Sonnet 3.5

#### Available Actions

**Basic Actions (all versions):**
- `screenshot` - Capture current display
- `left_click` - Click at coordinates
- `type` - Enter text string
- `key` - Press key combinations
- `mouse_move` - Move cursor to coordinates
- `left_click_drag` - Click and drag between coordinates
- `right_click`, `middle_click` - Additional mouse buttons
- `double_click` - Double-click action
- `cursor_position` - Get current cursor position

**Enhanced Actions (computer_20250124 only):**
- `scroll` - Directional scrolling with amount control
- `triple_click` - Triple-click action
- `left_mouse_down`, `left_mouse_up` - Fine-grained click control
- `hold_key` - Hold keys while performing other actions
- `wait` - Add pauses between actions

#### Configuration Options

- `--display-width` - Display width in pixels (default: 1024)
- `--display-height` - Display height in pixels (default: 768)
- `--display-number` - X11 display number for multi-display setups
- `--computer-tool-version` - Tool version selection

#### Beta Headers

The implementation automatically adds the correct beta headers based on:
- Model type (Claude 4, Sonnet 3.7, Sonnet 3.5)
- Tool version selected
- Uses `client.beta.messages.create()` when computer use tools are present

### Usage Examples

#### Basic Usage
```bash
# Enable computer use with default settings
claude-agent --interactive --tools computer_use

# Custom display size
claude-agent --interactive --tools computer_use --display-width 1280 --display-height 800

# Specific tool version for Claude Sonnet 3.5
claude-agent --interactive --tools computer_use --computer-tool-version computer_20241022
```

#### Example Prompts
- "Take a screenshot of the current desktop"
- "Click on the button at coordinates 100, 200"
- "Type 'Hello World' into the current window"
- "Press Ctrl+C to copy"
- "Scroll down 3 times in the current window"

### Security Considerations

The implementation includes several security features:
- Coordinate validation and bounds checking
- Command sanitization using `shlex.quote()`
- Duration limits for hold_key and wait actions
- Error handling for invalid actions

### Dependencies

- `anthropic` - Claude API client with beta support
- `mcp` - Model Context Protocol support
- Standard Python libraries for system interaction

### Platform Requirements

- Linux/Unix environment with X11 display server
- Screenshot tools: `gnome-screenshot`, `scrot`, or ImageMagick `import`
- Mouse/keyboard control: `xdotool`
- Image processing: ImageMagick `convert` (optional, for scaling)

### Testing

The implementation includes:
- Import validation
- Tool instantiation tests
- Schema validation
- Basic functionality verification

### Integration with Existing Framework

The computer use tool integrates seamlessly with the existing agent framework:
- Follows the same `Tool` base class pattern
- Uses async execution model
- Supports the same configuration and verbose logging
- Works with MCP servers and other tools

## Next Steps

1. **Environment Setup**: Set up a proper X11 environment for testing
2. **Docker Integration**: Create containerized environment for safe computer use
3. **Enhanced Error Handling**: Add more robust error recovery
4. **Performance Optimization**: Optimize screenshot capture and processing
5. **Additional Actions**: Implement any missing actions as needed

## Notes

- Computer use requires a graphical environment to function properly
- Screenshots are base64 encoded and included in tool responses
- The tool automatically scales coordinates for optimal performance
- Beta headers are automatically managed based on model and tool version
