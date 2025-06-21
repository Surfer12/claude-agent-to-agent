# Code Execution Tool Implementation

## Overview

We have successfully implemented the code execution tool for the Claude Agent-to-Agent CLI. This tool enables Claude to execute Python code in a secure, sandboxed environment for data analysis, visualization, calculations, and file processing.

## Implementation Details

### Files Created/Modified

1. **`agents/tools/code_execution.py`** - Main code execution tool implementation
2. **`cli.py`** - Updated to include code execution tool options
3. **`agents/agent.py`** - Modified to support beta headers for code execution
4. **`README.md`** - Updated with code execution documentation
5. **`examples/code_execution_example.py`** - Example usage script

### Key Features

#### Tool Classes

1. **`CodeExecutionTool`** - Basic code execution functionality
2. **`CodeExecutionWithFilesTool`** - Enhanced version with file upload support

#### Model Support

The tool automatically validates model compatibility:
- Claude Opus 4 (`claude-opus-4-20250514`)
- Claude Sonnet 4 (`claude-sonnet-4-20250514`)
- Claude Sonnet 3.7 (`claude-3-7-sonnet-20250219`)
- Claude Haiku 3.5 (`claude-3-5-haiku-latest`)

#### Beta Headers

The implementation automatically manages beta headers:
- `code-execution-2025-05-22` - Required for all code execution
- `files-api-2025-04-14` - Additional header for file support

### Configuration Options

- `--tools code_execution` - Enable basic code execution
- `--enable-file-support` - Enable file upload capabilities

### Usage Examples

#### Basic Usage
```bash
# Enable code execution
claude-agent --interactive --tools code_execution

# With file support
claude-agent --interactive --tools code_execution --enable-file-support

# Specific model
claude-agent --interactive --tools code_execution --model claude-opus-4-20250514
```

#### Example Prompts
- "Calculate the mean and standard deviation of [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]"
- "Create a matplotlib chart showing sales data over time"
- "Generate a random dataset and perform linear regression"
- "Analyze this CSV file and provide summary statistics"

### Runtime Environment

The code execution tool runs in Claude's secure sandbox with:
- **Python version**: 3.11.12
- **Memory**: 1GiB RAM
- **Disk space**: 5GiB workspace storage
- **CPU**: 1 CPU
- **Container lifetime**: 1 hour
- **Internet access**: Disabled for security

### Pre-installed Libraries

The sandbox includes commonly used Python libraries:
- **Data Science**: pandas, numpy, scipy, scikit-learn, statsmodels
- **Visualization**: matplotlib
- **File Processing**: pyarrow, openpyxl, xlrd, pillow
- **Math & Computing**: sympy, mpmath
- **Utilities**: tqdm, python-dateutil, pytz, joblib

### Security Features

- **Sandboxed execution**: Code runs in isolated containers
- **No internet access**: Prevents external data exfiltration
- **Resource limits**: Memory, CPU, and disk space constraints
- **Time limits**: Container expiration after 1 hour
- **Workspace isolation**: Scoped to API key workspace

### Integration with Existing Framework

The code execution tool integrates seamlessly:
- Follows the same `Tool` base class pattern
- Uses async execution model (server-side execution)
- Supports the same configuration and verbose logging
- Works with other tools and MCP servers
- Automatic beta header management

### Tool Schema

#### Basic Tool
```json
{
  "type": "code_execution_20250522",
  "name": "code_execution"
}
```

#### Input Schema
```json
{
  "type": "object",
  "properties": {
    "code": {
      "type": "string",
      "description": "Python code to execute in the sandbox environment"
    }
  },
  "required": ["code"]
}
```

### Response Format

Code execution results include:
- `stdout`: Output from print statements and successful execution
- `stderr`: Error messages if code execution fails
- `return_code`: 0 for success, non-zero for failure

### File Support

When file support is enabled:
- Upload files using the Files API
- Reference files in messages using `container_upload` content blocks
- Process CSV, Excel, JSON, XML, images, and text files
- Retrieve generated files (plots, processed data, etc.)

### Pricing

Code execution usage is tracked separately:
- **Pricing**: $0.05 per session-hour
- **Minimum billing**: 5 minutes per session
- **File preloading**: Billed even if tool isn't used when files are included

### Error Handling

The implementation handles various error scenarios:
- Model compatibility validation
- Beta header management
- Tool instantiation errors
- Server-side execution errors

### Testing

The implementation includes:
- Import validation
- Tool instantiation tests
- Model support validation
- Schema validation
- CLI integration tests

## Usage Patterns

### Data Analysis
```bash
claude-agent --prompt "Analyze the correlation between variables in this dataset" --tools code_execution --enable-file-support
```

### Visualization
```bash
claude-agent --prompt "Create a histogram of the data and save as PNG" --tools code_execution
```

### Mathematical Calculations
```bash
claude-agent --prompt "Solve this system of linear equations using numpy" --tools code_execution
```

### File Processing
```bash
claude-agent --prompt "Convert this Excel file to CSV format" --tools code_execution --enable-file-support
```

## Next Steps

1. **File Upload Integration**: Add CLI support for uploading files
2. **Container Management**: Implement container reuse for multi-turn conversations
3. **Output Handling**: Enhanced handling of generated files and visualizations
4. **Streaming Support**: Add support for streaming code execution results
5. **Custom Libraries**: Support for additional Python packages if needed

## Notes

- Code execution is handled server-side by Claude's API
- The tool implementation focuses on proper schema and beta header management
- File support requires additional beta headers and Files API integration
- Container state persists across requests within the same conversation
- Generated files can be retrieved using the Files API

This implementation provides a robust foundation for Python code execution within the Claude Agent-to-Agent CLI framework.
