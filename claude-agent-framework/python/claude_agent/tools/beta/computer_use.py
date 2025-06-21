"""Computer use tool for desktop interaction."""

import asyncio
import base64
import os
import shlex
import shutil
import subprocess
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal, TypedDict
from uuid import uuid4

from ..base import Tool

OUTPUT_DIR = "/tmp/outputs"
TYPING_DELAY_MS = 12
TYPING_GROUP_SIZE = 50

# Action types for different tool versions
Action_20241022 = Literal[
    "key",
    "type", 
    "mouse_move",
    "left_click",
    "left_click_drag",
    "right_click",
    "middle_click", 
    "double_click",
    "screenshot",
    "cursor_position",
]

Action_20250124 = Literal[
    "key",
    "type",
    "mouse_move", 
    "left_click",
    "left_click_drag",
    "right_click",
    "middle_click",
    "double_click", 
    "triple_click",
    "left_mouse_down",
    "left_mouse_up",
    "scroll",
    "hold_key",
    "wait",
    "screenshot",
    "cursor_position",
]

ScrollDirection = Literal["up", "down", "left", "right"]

class Resolution(TypedDict):
    width: int
    height: int

# Maximum scaling targets for optimal performance
MAX_SCALING_TARGETS: dict[str, Resolution] = {
    "XGA": Resolution(width=1024, height=768),   # 4:3
    "WXGA": Resolution(width=1280, height=800),  # 16:10
    "FWXGA": Resolution(width=1366, height=768), # ~16:9
}

CLICK_BUTTONS = {
    "left_click": 1,
    "right_click": 3,
    "middle_click": 2,
    "double_click": "--repeat 2 --delay 10 1",
    "triple_click": "--repeat 3 --delay 10 1",
}

class ScalingSource(StrEnum):
    COMPUTER = "computer"
    API = "api"


def chunks(s: str, chunk_size: int) -> list[str]:
    """Split string into chunks of specified size."""
    return [s[i : i + chunk_size] for i in range(0, len(s), chunk_size)]


async def run_command(command: str) -> tuple[int, str, str]:
    """Run a shell command asynchronously."""
    process = await asyncio.create_subprocess_shell(
        command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate()
    return (
        process.returncode or 0,
        stdout.decode() if stdout else "",
        stderr.decode() if stderr else "",
    )


class ComputerUseTool(Tool):
    """Computer use tool for desktop interaction via screenshots and input control."""
    
    def __init__(
        self,
        display_width_px: int = 1024,
        display_height_px: int = 768,
        display_number: int | None = None,
        tool_version: Literal["computer_20241022", "computer_20250124"] = "computer_20250124",
        scaling_enabled: bool = True,
        screenshot_delay: float = 2.0,
    ):
        """Initialize computer use tool.
        
        Args:
            display_width_px: Display width in pixels
            display_height_px: Display height in pixels  
            display_number: X11 display number (None for default)
            tool_version: Tool version to use
            scaling_enabled: Whether to enable coordinate scaling
            screenshot_delay: Delay before taking screenshots
        """
        self.width = display_width_px
        self.height = display_height_px
        self.display_num = display_number
        self.tool_version = tool_version
        self._scaling_enabled = scaling_enabled
        self._screenshot_delay = screenshot_delay
        
        # Set up display prefix for X11 commands
        if display_number is not None:
            self._display_prefix = f"DISPLAY=:{display_number} "
        else:
            self._display_prefix = ""
            
        self.xdotool = f"{self._display_prefix}xdotool"
        
        # Define tool schema based on version
        if tool_version == "computer_20250124":
            action_enum = [
                "key", "type", "mouse_move", "left_click", "left_click_drag",
                "right_click", "middle_click", "double_click", "triple_click",
                "left_mouse_down", "left_mouse_up", "scroll", "hold_key", 
                "wait", "screenshot", "cursor_position"
            ]
        else:
            action_enum = [
                "key", "type", "mouse_move", "left_click", "left_click_drag", 
                "right_click", "middle_click", "double_click", "screenshot",
                "cursor_position"
            ]
            
        input_schema = {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": action_enum,
                    "description": "The action to perform"
                },
                "text": {
                    "type": "string", 
                    "description": "Text to type or key combination to press"
                },
                "coordinate": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "minItems": 2,
                    "maxItems": 2,
                    "description": "X, Y coordinates for mouse actions"
                }
            },
            "required": ["action"]
        }
        
        # Add version-specific properties
        if tool_version == "computer_20250124":
            input_schema["properties"].update({
                "scroll_direction": {
                    "type": "string",
                    "enum": ["up", "down", "left", "right"],
                    "description": "Direction to scroll"
                },
                "scroll_amount": {
                    "type": "integer",
                    "minimum": 0,
                    "description": "Amount to scroll"
                },
                "duration": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 100,
                    "description": "Duration in seconds for hold_key or wait actions"
                },
                "key": {
                    "type": "string",
                    "description": "Modifier key to hold during click actions"
                }
            })
        
        super().__init__(
            name="computer",
            description="A tool for interacting with the desktop environment through screenshots, mouse control, and keyboard input.",
            input_schema=input_schema,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert tool to Claude API format with beta tool parameters."""
        return {
            "type": self.tool_version,
            "name": self.name,
            "display_width_px": self.width,
            "display_height_px": self.height,
            "display_number": self.display_num,
        }

    def validate_and_get_coordinates(self, coordinate: list[int] | None = None) -> tuple[int, int]:
        """Validate and scale coordinates."""
        if not isinstance(coordinate, list) or len(coordinate) != 2:
            raise ValueError(f"{coordinate} must be a list of length 2")
        if not all(isinstance(i, int) and i >= 0 for i in coordinate):
            raise ValueError(f"{coordinate} must be a list of non-negative ints")
            
        return self.scale_coordinates(ScalingSource.API, coordinate[0], coordinate[1])

    def scale_coordinates(self, source: ScalingSource, x: int, y: int) -> tuple[int, int]:
        """Scale coordinates to target resolution if scaling is enabled."""
        if not self._scaling_enabled:
            return x, y
            
        ratio = self.width / self.height
        target_dimension = None
        
        for dimension in MAX_SCALING_TARGETS.values():
            # Allow some error in aspect ratio
            if abs(dimension["width"] / dimension["height"] - ratio) < 0.02:
                if dimension["width"] < self.width:
                    target_dimension = dimension
                break
                
        if target_dimension is None:
            return x, y
            
        x_scaling_factor = target_dimension["width"] / self.width
        y_scaling_factor = target_dimension["height"] / self.height
        
        if source == ScalingSource.API:
            if x > self.width or y > self.height:
                raise ValueError(f"Coordinates {x}, {y} are out of bounds")
            # Scale up
            return round(x / x_scaling_factor), round(y / y_scaling_factor)
        
        # Scale down
        return round(x * x_scaling_factor), round(y * y_scaling_factor)

    async def take_screenshot(self) -> str:
        """Take a screenshot and return base64 encoded image."""
        output_dir = Path(OUTPUT_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"screenshot_{uuid4().hex}.png"
        
        # Try different screenshot tools
        screenshot_cmd = None
        if shutil.which("gnome-screenshot"):
            screenshot_cmd = f"{self._display_prefix}gnome-screenshot -f {path} -p"
        elif shutil.which("scrot"):
            screenshot_cmd = f"{self._display_prefix}scrot -p {path}"
        elif shutil.which("import"):  # ImageMagick
            screenshot_cmd = f"{self._display_prefix}import -window root {path}"
        else:
            raise RuntimeError("No screenshot tool available (gnome-screenshot, scrot, or ImageMagick)")
            
        returncode, stdout, stderr = await run_command(screenshot_cmd)
        
        if returncode != 0:
            raise RuntimeError(f"Screenshot failed: {stderr}")
            
        # Scale image if needed
        if self._scaling_enabled:
            x, y = self.scale_coordinates(ScalingSource.COMPUTER, self.width, self.height)
            if shutil.which("convert"):  # ImageMagick
                await run_command(f"convert {path} -resize {x}x{y}! {path}")
                
        if path.exists():
            return base64.b64encode(path.read_bytes()).decode()
        else:
            raise RuntimeError("Screenshot file was not created")

    async def shell_command(self, command: str, take_screenshot: bool = True) -> dict[str, Any]:
        """Execute shell command and optionally take screenshot."""
        returncode, stdout, stderr = await run_command(command)
        
        result = {
            "output": stdout,
            "error": stderr,
            "returncode": returncode
        }
        
        if take_screenshot:
            await asyncio.sleep(self._screenshot_delay)
            try:
                result["base64_image"] = await self.take_screenshot()
            except Exception as e:
                result["error"] = f"{stderr}\nScreenshot error: {str(e)}"
                
        return result

    async def execute(self, **kwargs) -> str:
        """Execute computer use action."""
        action = kwargs.get("action")
        text = kwargs.get("text")
        coordinate = kwargs.get("coordinate")
        
        if not action:
            return "Error: action parameter is required"
            
        try:
            # Handle mouse movement and drag actions
            if action in ("mouse_move", "left_click_drag"):
                if coordinate is None:
                    return f"Error: coordinate is required for {action}"
                if text is not None:
                    return f"Error: text is not accepted for {action}"
                    
                x, y = self.validate_and_get_coordinates(coordinate)
                
                if action == "mouse_move":
                    command = f"{self.xdotool} mousemove --sync {x} {y}"
                elif action == "left_click_drag":
                    command = f"{self.xdotool} mousedown 1 mousemove --sync {x} {y} mouseup 1"
                    
                result = await self.shell_command(command)
                return self._format_result(result)
                
            # Handle keyboard actions
            elif action in ("key", "type"):
                if text is None:
                    return f"Error: text is required for {action}"
                if coordinate is not None:
                    return f"Error: coordinate is not accepted for {action}"
                if not isinstance(text, str):
                    return f"Error: text must be a string"
                    
                if action == "key":
                    command = f"{self.xdotool} key -- {shlex.quote(text)}"
                    result = await self.shell_command(command)
                    return self._format_result(result)
                    
                elif action == "type":
                    # Type in chunks to avoid issues with long text
                    results = []
                    for chunk in chunks(text, TYPING_GROUP_SIZE):
                        command = f"{self.xdotool} type --delay {TYPING_DELAY_MS} -- {shlex.quote(chunk)}"
                        chunk_result = await self.shell_command(command, take_screenshot=False)
                        results.append(chunk_result)
                        
                    # Take final screenshot
                    screenshot = await self.take_screenshot()
                    combined_output = "".join(r.get("output", "") for r in results)
                    combined_error = "".join(r.get("error", "") for r in results)
                    
                    return self._format_result({
                        "output": combined_output,
                        "error": combined_error,
                        "base64_image": screenshot
                    })
                    
            # Handle click actions
            elif action in ("left_click", "right_click", "double_click", "middle_click"):
                if text is not None:
                    return f"Error: text is not accepted for {action}"
                    
                # Handle coordinate-based clicks
                if coordinate is not None:
                    x, y = self.validate_and_get_coordinates(coordinate)
                    move_cmd = f"mousemove --sync {x} {y}"
                else:
                    move_cmd = ""
                    
                # Handle modifier keys for enhanced version
                key_modifier = kwargs.get("key")
                key_down = f"keydown {key_modifier}" if key_modifier else ""
                key_up = f"keyup {key_modifier}" if key_modifier else ""
                
                click_cmd = f"click {CLICK_BUTTONS[action]}"
                command_parts = [self.xdotool, move_cmd, key_down, click_cmd, key_up]
                command = " ".join(filter(None, command_parts))
                
                result = await self.shell_command(command)
                return self._format_result(result)
                
            # Handle enhanced actions (version 20250124)
            elif action == "triple_click" and self.tool_version == "computer_20250124":
                if coordinate is not None:
                    x, y = self.validate_and_get_coordinates(coordinate)
                    move_cmd = f"mousemove --sync {x} {y}"
                else:
                    move_cmd = ""
                    
                key_modifier = kwargs.get("key")
                key_down = f"keydown {key_modifier}" if key_modifier else ""
                key_up = f"keyup {key_modifier}" if key_modifier else ""
                
                click_cmd = f"click {CLICK_BUTTONS[action]}"
                command_parts = [self.xdotool, move_cmd, key_down, click_cmd, key_up]
                command = " ".join(filter(None, command_parts))
                
                result = await self.shell_command(command)
                return self._format_result(result)
                
            elif action in ("left_mouse_down", "left_mouse_up") and self.tool_version == "computer_20250124":
                if coordinate is not None:
                    return f"Error: coordinate is not accepted for {action}"
                    
                mouse_action = "mousedown" if action == "left_mouse_down" else "mouseup"
                command = f"{self.xdotool} {mouse_action} 1"
                result = await self.shell_command(command)
                return self._format_result(result)
                
            elif action == "scroll" and self.tool_version == "computer_20250124":
                scroll_direction = kwargs.get("scroll_direction")
                scroll_amount = kwargs.get("scroll_amount", 3)
                
                if scroll_direction not in ["up", "down", "left", "right"]:
                    return "Error: scroll_direction must be 'up', 'down', 'left', or 'right'"
                if not isinstance(scroll_amount, int) or scroll_amount < 0:
                    return "Error: scroll_amount must be a non-negative integer"
                    
                # Move to coordinate if specified
                move_cmd = ""
                if coordinate is not None:
                    x, y = self.validate_and_get_coordinates(coordinate)
                    move_cmd = f"mousemove --sync {x} {y}"
                    
                scroll_button = {"up": 4, "down": 5, "left": 6, "right": 7}[scroll_direction]
                
                # Handle modifier keys
                key_modifier = kwargs.get("text")  # text field used for modifier keys in scroll
                key_down = f"keydown {key_modifier}" if key_modifier else ""
                key_up = f"keyup {key_modifier}" if key_modifier else ""
                
                scroll_cmd = f"click --repeat {scroll_amount} {scroll_button}"
                command_parts = [self.xdotool, move_cmd, key_down, scroll_cmd, key_up]
                command = " ".join(filter(None, command_parts))
                
                result = await self.shell_command(command)
                return self._format_result(result)
                
            elif action == "hold_key" and self.tool_version == "computer_20250124":
                if text is None:
                    return "Error: text is required for hold_key action"
                    
                duration = kwargs.get("duration", 1.0)
                if not isinstance(duration, (int, float)) or duration < 0:
                    return "Error: duration must be a non-negative number"
                if duration > 100:
                    return "Error: duration is too long (max 100 seconds)"
                    
                escaped_keys = shlex.quote(text)
                command = f"{self.xdotool} keydown {escaped_keys} sleep {duration} keyup {escaped_keys}"
                result = await self.shell_command(command)
                return self._format_result(result)
                
            elif action == "wait" and self.tool_version == "computer_20250124":
                duration = kwargs.get("duration", 1.0)
                if not isinstance(duration, (int, float)) or duration < 0:
                    return "Error: duration must be a non-negative number"
                if duration > 100:
                    return "Error: duration is too long (max 100 seconds)"
                    
                await asyncio.sleep(duration)
                screenshot = await self.take_screenshot()
                return self._format_result({"base64_image": screenshot})
                
            # Handle screenshot action
            elif action == "screenshot":
                if text is not None or coordinate is not None:
                    return "Error: screenshot action does not accept text or coordinate parameters"
                    
                screenshot = await self.take_screenshot()
                return self._format_result({"base64_image": screenshot})
                
            # Handle cursor position
            elif action == "cursor_position":
                if text is not None or coordinate is not None:
                    return "Error: cursor_position action does not accept text or coordinate parameters"
                    
                command = f"{self.xdotool} getmouselocation --shell"
                result = await self.shell_command(command, take_screenshot=False)
                
                if result.get("output"):
                    output = result["output"]
                    try:
                        x = int(output.split("X=")[1].split("\n")[0])
                        y = int(output.split("Y=")[1].split("\n")[0])
                        scaled_x, scaled_y = self.scale_coordinates(ScalingSource.COMPUTER, x, y)
                        return f"Cursor position: X={scaled_x}, Y={scaled_y}"
                    except (IndexError, ValueError):
                        return f"Error parsing cursor position: {output}"
                else:
                    return f"Error getting cursor position: {result.get('error', 'Unknown error')}"
                    
            else:
                return f"Error: Invalid action '{action}' for tool version {self.tool_version}"
                
        except Exception as e:
            return f"Error executing {action}: {str(e)}"

    def _format_result(self, result: dict[str, Any]) -> str:
        """Format tool execution result."""
        output_parts = []
        
        if result.get("output"):
            output_parts.append(f"Output: {result['output'].strip()}")
            
        if result.get("error"):
            output_parts.append(f"Error: {result['error'].strip()}")
            
        if result.get("base64_image"):
            output_parts.append("Screenshot captured successfully")
            
        if result.get("returncode", 0) != 0:
            output_parts.append(f"Command returned exit code: {result['returncode']}")
            
        return "\n".join(output_parts) if output_parts else "Action completed successfully"
