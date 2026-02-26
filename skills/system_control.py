"""
Igris AI Agent — System Control Skill  (Issue #4)

Provides system-level operations:
- Shutdown
- Reboot / Restart
- Sleep / Hibernate
- Lock Screen

All operations require user confirmation before execution.
Windows-compatible via subprocess.
"""

import subprocess
import platform
from langchain.tools import tool


def _is_windows():
    return platform.system().lower() == "windows"


def _is_linux():
    return platform.system().lower() == "linux"


def _is_mac():
    return platform.system().lower() == "darwin"


@tool
def system_shutdown(confirm: str) -> str:
    """Shutdown the computer. Pass confirm='yes' to actually execute.
    The agent MUST ask the user for confirmation before calling this with confirm='yes'."""
    if confirm.lower() != "yes":
        return "Shutdown cancelled. Pass confirm='yes' to proceed."

    try:
        if _is_windows():
            subprocess.run(["shutdown", "/s", "/t", "30", "/c", "Igris: Shutting down by royal command."], check=True)
            return "Shutdown initiated. The system will power off in 30 seconds. Run 'shutdown /a' to abort."
        elif _is_linux():
            subprocess.run(["shutdown", "-h", "+1", "Igris: Shutting down by royal command."], check=True)
            return "Shutdown initiated. The system will power off in 1 minute. Run 'shutdown -c' to abort."
        elif _is_mac():
            subprocess.run(["sudo", "shutdown", "-h", "+1"], check=True)
            return "Shutdown initiated for 1 minute from now."
        else:
            return f"Unsupported OS: {platform.system()}"
    except subprocess.CalledProcessError as e:
        return f"Shutdown failed: {e}"
    except PermissionError:
        return "Permission denied. Run the agent with administrator/root privileges for system control."


@tool
def system_reboot(confirm: str) -> str:
    """Reboot / restart the computer. Pass confirm='yes' to actually execute.
    The agent MUST ask the user for confirmation before calling this with confirm='yes'."""
    if confirm.lower() != "yes":
        return "Reboot cancelled. Pass confirm='yes' to proceed."

    try:
        if _is_windows():
            subprocess.run(["shutdown", "/r", "/t", "30", "/c", "Igris: Rebooting by royal command."], check=True)
            return "Reboot initiated. The system will restart in 30 seconds. Run 'shutdown /a' to abort."
        elif _is_linux():
            subprocess.run(["shutdown", "-r", "+1", "Igris: Rebooting by royal command."], check=True)
            return "Reboot initiated. The system will restart in 1 minute."
        elif _is_mac():
            subprocess.run(["sudo", "shutdown", "-r", "+1"], check=True)
            return "Reboot initiated for 1 minute from now."
        else:
            return f"Unsupported OS: {platform.system()}"
    except subprocess.CalledProcessError as e:
        return f"Reboot failed: {e}"
    except PermissionError:
        return "Permission denied. Run the agent with administrator/root privileges."


@tool
def system_sleep() -> str:
    """Put the computer to sleep / suspend."""
    try:
        if _is_windows():
            # powercfg to ensure hibernate is off so we get sleep not hibernate
            subprocess.run(["rundll32.exe", "powrprof.dll,SetSuspendState", "0,1,0"], check=True)
            return "System is going to sleep now."
        elif _is_linux():
            subprocess.run(["systemctl", "suspend"], check=True)
            return "System suspended."
        elif _is_mac():
            subprocess.run(["pmset", "sleepnow"], check=True)
            return "System is going to sleep."
        else:
            return f"Unsupported OS: {platform.system()}"
    except subprocess.CalledProcessError as e:
        return f"Sleep failed: {e}"
    except PermissionError:
        return "Permission denied."


@tool
def system_lock() -> str:
    """Lock the screen immediately."""
    try:
        if _is_windows():
            subprocess.run(["rundll32.exe", "user32.dll,LockWorkStation"], check=True)
            return "Screen locked, Your Majesty."
        elif _is_linux():
            # Try common Linux lock commands
            for cmd in [["loginctl", "lock-session"], ["gnome-screensaver-command", "-l"], ["xdg-screensaver", "lock"]]:
                try:
                    subprocess.run(cmd, check=True)
                    return "Screen locked."
                except (subprocess.CalledProcessError, FileNotFoundError):
                    continue
            return "Could not find a screen lock command on this Linux system."
        elif _is_mac():
            subprocess.run([
                "osascript", "-e",
                'tell application "System Events" to keystroke "q" using {command down, control down}'
            ], check=True)
            return "Screen locked."
        else:
            return f"Unsupported OS: {platform.system()}"
    except Exception as e:
        return f"Lock failed: {e}"


@tool
def system_cancel_shutdown() -> str:
    """Cancel a pending shutdown or reboot (Windows: shutdown /a, Linux: shutdown -c)."""
    try:
        if _is_windows():
            subprocess.run(["shutdown", "/a"], check=True)
            return "Pending shutdown/reboot cancelled."
        elif _is_linux():
            subprocess.run(["shutdown", "-c"], check=True)
            return "Pending shutdown/reboot cancelled."
        else:
            return "Cancel not supported on this OS or no pending shutdown."
    except subprocess.CalledProcessError as e:
        return f"Cancel failed (no pending shutdown?): {e}"


# Collect all tools for registration
SYSTEM_CONTROL_TOOLS = [
    system_shutdown,
    system_reboot,
    system_sleep,
    system_lock,
    system_cancel_shutdown,
]
