       #!/usr/bin/env python3
# timeseries-api/utilities/chronicler.py

import logging
import json
import time
import os
import sys
import subprocess
from typing import List, Dict, Union, Optional

# colorful logging must be imported and initialized before any other logging scripts
from colorama import init
init(autoreset=True)


class JsonFormatter(logging.Formatter):
    """Format logs as JSON objects for better cloud integration."""
    
    def format(self, record):
        log_record = {
            "timestamp": self.formatTime(record, "%Y%m%d_%H%M%S"),
            "level": record.levelname,
            "filename": record.filename,
            "line": record.lineno,
            "function": record.funcName,
            "message": record.getMessage(),
        }
        
        # Add exception info if present
        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)
            
        # Add extra fields if present
        if hasattr(record, "extra_fields"):
            log_record.update(record.extra_fields)
            
        return json.dumps(log_record)

class GitInfo:
    """
    A class to retrieve Git branch, commit hash, and repo state.
    """

    def __init__(self, repo_path: str = "./") -> None:
        """
        Initializes the GitInfo class.

        Args:
            repo_path (str, optional): Path to the Git repository. Defaults to "./".
        """
        self.repo_path = repo_path
        self.branch: Optional[str] = None
        self.commit_hash: Optional[str] = None
        self.is_clean: Optional[Union[str, bool]] = None
        self.update_git_info()

    def run_git_command(self, command: List[str]) -> str:
        """
        Runs a Git command in the repository.

        Args:
            command (List[str]): A list of command arguments.

        Returns:
            str: Output from the Git command or an error message if the command fails.
        """
        try:
            return subprocess.check_output(command, cwd=self.repo_path, shell=False).strip().decode()  # shell False to avoid shell injection
        except subprocess.CalledProcessError:
            return "Not a repo"
        except FileNotFoundError:
            return "Not a repo"

    def update_git_info(self) -> None:
        """
        Updates the Git branch, commit hash, and repository state.
        """
        self.branch = self.run_git_command(["git", "rev-parse", "--abbrev-ref", "HEAD"])
        self.commit_hash = self.run_git_command(["git", "rev-parse", "--short", "HEAD"])
        status_output = self.run_git_command(["git", "status", "--porcelain"])

        if status_output == "Not a repo":
            self.is_clean = "Not a repo"
        elif status_output:
            self.is_clean = False
        else:
            self.is_clean = True

    def get_info(self) -> Dict[str, Union[str, bool, None]]:
        """
        Returns Git information as a dictionary.

        Returns:
            Dict[str, Union[str, bool, None]]: Git information including branch, commit hash, and cleanliness status.
        """
        return {
            "branch": self.branch,
            "commit_hash": self.commit_hash,
            "is_clean": self.is_clean,
        }


class Chronicler:
    """
    A logging utility class to initialize and manage logging configuration.

    Writes logs to both `stdout` (for AWS CloudWatch) and a timestamped log file in `./logs`.
    """

    def __init__(self, script_path: str, use_json: bool = False) -> None:
        """
        Initializes the Chronicler class and sets up logging for the script.

        Args:
            script_path (str): Path of the script for which logging is being initialized.
            use_json (bool): Whether to use JSON formatting for logs (useful for cloud environments).
        """
        script_name = os.path.splitext(os.path.basename(script_path))[0]
        timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        self.log_file = f"./logs/{timestamp}_{script_name}.log"
        os.makedirs("./logs", exist_ok=True)

        # Create handlers
        stdout_handler = logging.StreamHandler(sys.stdout)
        file_handler = logging.FileHandler(filename=self.log_file, mode="w")
        
        # Apply formatting based on configuration
        if use_json:
            json_formatter = JsonFormatter()
            stdout_handler.setFormatter(json_formatter)
            file_handler.setFormatter(json_formatter)
        else:
            # Traditional format
            log_format = "%(asctime)s %(levelname)s %(filename)s:%(lineno)d %(funcName)s| %(message)s"
            formatter = logging.Formatter(log_format, datefmt="%Y%m%d_%H%M%S")
            stdout_handler.setFormatter(formatter)
            file_handler.setFormatter(formatter)
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        root_logger.addHandler(stdout_handler)
        root_logger.addHandler(file_handler)
        
        ascii_banner = """\n
        >  ┓       • ┓     <
        > ┏┣┓┏┓┏┓┏┓┓┏┃┏┓┏┓ <
        > ┗┛┗┛ ┗┛┛┗┗┗┗┗ ┛  <
        """
        logging.info(ascii_banner)
        logging.info(f"Logging initialized for {script_path}")

        git_info = GitInfo(repo_path="./")
        git_meta = git_info.get_info()
        logging.info(f"git branch: {git_meta['branch']}")
        logging.info(f"git rev-parse HEAD: {git_meta['commit_hash']}")
        logging.info(
            f"git status --porcelain: {'is_clean' if git_meta['is_clean'] else 'is_dirty'}"
        )


def init_chronicler(use_json: bool = False) -> Chronicler:
    """
    Initializes and returns an instance of the Chronicler class.
    
    Args:
        use_json (bool): Whether to use JSON formatted logs (recommended for cloud environments).

    Returns:
        Chronicler: An instance of the Chronicler class.
    """
    current_script_path = os.path.abspath(__file__)
    return Chronicler(current_script_path, use_json=use_json)
