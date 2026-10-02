import asyncio
import asyncssh
from asyncssh import SFTPClient, SFTPError
import logging
import os
from pathlib import Path
import posixpath
import sys
import socket
from datetime import datetime
import time
import glob
from typing import Union
import argparse
import traceback
if sys.version_info >= (3, 8):
    from typing import TypedDict  # pylint: disable=no-name-in-module
else:
    from typing_extensions import TypedDict

def is_connected(host="8.8.8.8", port=53, timeout=3):
    """
    Check if the device is connected to the internet by attempting a socket connection.
    Args:
        host (str): The host to connect to. Default is Google's public DNS server.
        port (int): The port to use. Default is 53 (DNS).
        timeout (int): Connection timeout in seconds.
    Returns:
        bool: True if connected, False otherwise.
    """
    try:
        socket.setdefaulttimeout(timeout)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
        return True
    except socket.error:
        return False

def wait_for_connection(wait=5, timeout=30):
    """
    Wait until an internet connection is available.
    Args:
        wait (int): Waiting time before retrying.
        timeout (int): Connection timeout in seconds.
    Returns:
        bool: True if connected, False otherwise.
    """
    print("Waiting for WiFi connection...")
    total_wait = 0
    while total_wait < timeout and not is_connected():
        print("No connection. Retrying in 5 seconds...")
        time.sleep(wait)
        total_wait += wait
    if total_wait >= timeout:
        print("Failed to connect to the internet.")
        return False
    else:
        print("WiFi connected!")
        return True

def restart_wifi(wait=5):
    """Restart the WiFi interface."""
    print("Restarting WiFi...")
    os.system("sudo ifconfig wlan0 down")
    time.sleep(wait)
    os.system("sudo ifconfig wlan0 up")
    time.sleep(wait)

def is_valid_yyyymmdd(date_str: str) -> bool:
    """Check if the string is a date with format "YYYYMMDD"
    """
    # Check that the string has exactly 8 digits
    if len(date_str) != 8 or not date_str.isdigit():
        return False
    try:
        # Try parsing it as YYYYMMDD
        datetime.strptime(date_str, "%Y%m%d")
        return True
    except ValueError:
        # Raised if the date is invalid (e.g., 20251301)
        return False

class AsyncSFTPParams(TypedDict):
    host: str
    port: int
    username: str
    password: str
    client_keys: list[str]

class AsyncSFTP:
    """Handle connection and file transfer to a SFTP server.

    Parameters
    ----------
    sftp_params : AsyncSFTPParams
        Set of parameters to connect to the SFTP server.
    max_concurrent : int, default=16
        Maximum number of coroutines to upload files on server.
    log_file : int, default=None
        Where to store the logs. If None, no logs will be stored.
    """
    def __init__(self, sftp_params: AsyncSFTPParams, max_concurrent: int = 16, log_file: str = None):
        self.sftp_params = sftp_params
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.to_execute = []

        self.missing_files = []
        self.nb_missing_files = 0

        # logging
        if log_file is not None:
            self.do_log = True
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.INFO)
            if not self.logger.handlers:
                file_handler = logging.FileHandler(log_file)
                file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s: %(message)s"))
                self.logger.addHandler(file_handler)
                self.logger.propagate = False  # Prevent messages from propagating to the root logger
                # Stream Handler to log to console
                stream_handler = logging.StreamHandler()
                stream_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s: %(message)s"))
                self.logger.addHandler(stream_handler)
        else:
            self.do_log = False

    async def _connect(self):
        """Establish a connection and return an SFTP client."""
        conn = await asyncssh.connect(**self.sftp_params)
        sftp = await conn.start_sftp_client()
        return conn, sftp

    def __getattr__(self, name):
        """Intercept undefined methods and proxy them to the SFTP client."""
        async def _proxy_method(*args, **kwargs):
            conn, sftp = await self._connect()
            try:
                # Get the requested attribute (like sftp.listdir)
                func = getattr(sftp, name)
                if not callable(func):
                    return func  # attribute (not method)
                # Call it with provided args/kwargs
                result = await func(*args, **kwargs)
                return result
            finally:
                conn.close()
                await conn.wait_closed()

        # Return a *synchronous* wrapper that runs the async proxy
        def _runner(*args, **kwargs):
            return asyncio.run(_proxy_method(*args, **kwargs))

        return _runner

    async def _put(self, sftp: SFTPClient, local_path: Path, remote_path: Path):
        """
        Read local file in a worker thread (Windows-safe) and write it to remote via asyncssh.
        """
        async with self.semaphore:
            try:
                local_path = Path(local_path)
                remote_path = str(remote_path)

                # Basic checks
                if not local_path.exists() or not local_path.is_file():
                    if self.do_log:
                        self.logger.error(f"Local file not found or not a file: {local_path}")
                    return

                local_size = local_path.stat().st_size
                if local_size == 0:
                    if self.do_log:
                        self.logger.warning(f"Skipping empty file: {local_path}")
                    return

                if self.do_log:
                    self.logger.debug(f"Reading local file {local_path} ({local_size} bytes) in thread...")

                loop = asyncio.get_running_loop()
                # Read the entire file in a thread to avoid Windows async file-I/O issues
                data = await loop.run_in_executor(None, local_path.read_bytes)

                if self.do_log:
                    self.logger.debug(f"Read {len(data)} bytes from {local_path}. Preparing remote file {remote_path}...")

                # Ensure remote parent directory exists
                parent = posixpath.dirname(remote_path)
                if parent:
                    await sftp.makedirs(parent, exist_ok=True)

                # Open remote file in binary write mode and write the bytes
                async with sftp.open(remote_path, pflags_or_mode='wb') as rf:
                    await rf.write(data)

                if self.do_log:
                    self.logger.debug(f"Uploaded {local_path} -> {remote_path} ({len(data)} bytes).")

            except Exception as e:
                if self.do_log:
                    self.logger.error(f"Failed to upload {local_path} to {remote_path}: {e}\n{traceback.format_exc()}")

    async def _mput(self, sftp: SFTPClient, localpaths: Union[list[str], str, Path], remotedir: Union[str, Path]):
        try:
            # Normalize inputs
            if isinstance(localpaths, (str, Path)):
                if os.path.isdir(str(localpaths)):
                    localpaths = [str(Path(localpaths) / f) for f in os.listdir(str(localpaths))]
                else:
                    localpaths = glob.glob(str(localpaths))
            elif not isinstance(localpaths, list):
                if self.do_log:
                    self.logger.error(f"Local paths must be either a list or a str, but found {type(localpaths)}.")
                return

            remotedir = Path(remotedir).as_posix()

            if len(localpaths) == 0:
                if self.do_log:
                    self.logger.error(f"No file found at {localpaths}.")
                return

            if self.do_log:
                self.logger.info(f"Creating/ensuring remote folder {remotedir}")
            await sftp.makedirs(remotedir, exist_ok=True)

            tasks = []
            for lp in localpaths:
                lp = Path(lp)
                remote_file = posixpath.join(remotedir, lp.name)
                if lp.is_dir():
                    # Recurse into subdirectory, mirror structure on remote
                    child_remote = posixpath.join(remotedir, lp.name)
                    tasks.append(self._mput(sftp, lp, child_remote))
                elif lp.is_file():
                    tasks.append(self._put(sftp, lp, remote_file))
                else:
                    if self.do_log:
                        self.logger.error(f"Local path contains an unknown file format {lp}.")
            if tasks:
                await asyncio.gather(*tasks)
        except Exception as e:
            if self.do_log:
                self.logger.error(f"Failed to upload files: {e}\n{traceback.format_exc()}")

    async def _put_folder(self, sftp: SFTPClient, local_folder: Path, remote_folder: Path):
        try:
            self.logger.info(f"Putting local folder {local_folder} to remote folder {remote_folder}.")
            await sftp.makedirs(remote_folder, exist_ok=True)
            tasks = []
            for f in os.listdir(local_folder):
                local_path = local_folder / f
                remote_path = remote_folder / f
                if os.path.isdir(local_path):
                    tasks.append(self._put_folder(sftp, local_path, remote_path))
                else:
                    tasks.append(self._put_file(sftp, local_path, remote_path))
            await asyncio.gather(*tasks)
            self.logger.info(f"Folder copied.")
        except Exception as e:
            if self.do_log:
                self.logger.error(f"Failed to upload folder {local_folder} to {remote_folder}: {e}")

    async def _save_listdir(self, sftp: SFTPClient, remote_folder: Path, filenames_remote_folder: Path):
        try:
            # Create filename folder if it does not exist
            await sftp.makedirs(filenames_remote_folder, exist_ok=True)

            # Get the list of files from the remote folder
            listdir = await sftp.listdir(remote_folder)

            # Name the output file after the remote folder name
            output_file = filenames_remote_folder / (remote_folder.name + ".txt")

            # Open the output file in writing mode and store the listdir into it
            async with sftp.open(output_file, pflags_or_mode='w') as f:
                await f.write('\n'.join(listdir))
            self.logger.info(f"Saved filenames from {remote_folder} to {output_file}.")
        except Exception as e:
            if self.do_log:
                self.logger.error(f"Failed to save filenames from {remote_folder} to {output_file}: {e}")

    async def _exec(self):
        async with asyncssh.connect(**self.sftp_params) as conn:
            async with conn.start_sftp_client() as sftp:
                if self.do_log:
                    self.logger.info(f"Established connection to SFTPServer.")
                start_time = time.time()
                for fn, args, kwargs in self.to_execute:
                    await fn(sftp, *args, **kwargs)

                # Monitor time
                end_time = time.time()
                duration = end_time - start_time
                if self.do_log:
                    self.logger.info(f"Execution time: {duration}")

                # Reset self.to_execute
                self.to_execute = []
    
    async def _list_missing_files(
        self,
        sftp: SFTPClient,
        local_folder: Path,
        remote_folder: Path,
        recursive: bool = False
    ):
        """
        Compare a local folder and a remote folder, returning files that exist
        locally but are missing remotely.

        Parameters
        ----------
        sftp : SFTPClient
            Active SFTP connection.
        local_folder : Path
            Path to the local folder to check.
        remote_folder : Path
            Path to the remote folder on the SFTP server.
        recursive : bool, default=False
            If True, check all subdirectories recursively.
        """

        try:
            # Try to list remote folder; if it doesn’t exist, assume empty
            try:
                if await sftp.exists(remote_folder.as_posix()) and await sftp.isdir(remote_folder.as_posix()):
                    remote_items = await sftp.listdir(remote_folder.as_posix())
                else:
                    self.logger.warning(f"The following path does not exists or is not a folder: {remote_folder}")
                    raise FileNotFoundError
            except FileNotFoundError:
                remote_items = []

            remote_items_set = set(remote_items)

            for item in os.listdir(local_folder):
                local_path = local_folder / item
                remote_path = remote_folder / item

                if local_path.is_file():
                    # File missing remotely → add to list
                    if item not in remote_items_set:
                        self.missing_files.append(str(local_path))
                elif recursive and local_path.is_dir():
                    # Only recurse if recursive=True
                    sub_missing = await self._list_missing_files(
                        sftp, local_path, remote_path, recursive=True
                    )
                    self.missing_files.extend(sub_missing)

        except Exception as e:
            if self.do_log:
                self.logger.error(
                    f"Error while comparing {local_folder} with {remote_folder}: {e}"
                )

    async def _put_missing_files(self, sftp: SFTPClient, local_folder: Path, remote_folder: Path):
        """
        Upload only the files that are present locally but missing remotely.
        """
        await self._list_missing_files(sftp, local_folder, remote_folder)

        if self.do_log:
            self.logger.info(f"Found {len(self.missing_files)} missing files to upload when comparing {local_folder} to {remote_folder}.")

        # Counter for missing files
        self.nb_missing_files += len(self.missing_files)

        if len(self.missing_files) > 0:
            await self._mput(sftp, self.missing_files, remote_folder)

            self.missing_files = []

    def list_missing_files(
        self,
        local_folder: str,
        remote_folder: str,
        recursive: bool = False,
        run: bool = False
    ) -> list[Path]:
        """
        Public method to list missing files between local and remote folders.

        Parameters
        ----------
        local_folder : str
            Path to the local folder.
        remote_folder : str
            Path to the remote folder on the SFTP server.
        recursive : bool, default=False
            If True, check all subdirectories recursively.
        run : bool, default=True
            If True, executes immediately and returns the missing files.

        Returns
        -------
        list[Path]
            List of local file paths missing remotely.
        """
        self.to_execute.append((self._list_missing_files, (Path(local_folder), Path(remote_folder)),{"recursive": recursive}))
        if run: self.run()

    def put_missing_files(self, local_folder: str, remote_folder: str, run: bool = False):
        """
        Public method to upload only the missing files between local and remote folders.
        """
        self.to_execute.append((self._put_missing_files, (Path(local_folder), Path(remote_folder)), {}))
        if run: self.run()

    def mput(self, localpaths: Union[str, list[str]], remotedir: str, run: bool = False):
        """Put local file to remote file.
        """
        self.to_execute.append((self._mput, (localpaths, Path(remotedir)), {}))
        if run: self.run()

    def put_file(self, local_path: str, remote_path: str, run: bool = False):
        """Put local file to remote file.
        """
        self.to_execute.append((self._put_file, (Path(local_path), Path(remote_path)), {}))
        if run: self.run()

    def put_folder(self, local_folder: str, remote_folder: str, run: bool = False):
        """Put local folder to remote folder. 
        Is recursive, so it works with nested folders.
        Create the remote folder if absent.
        """
        self.to_execute.append((self._put_folder, (Path(local_folder), Path(remote_folder)), {}))
        if run: self.run()

    def save_listdir(self, remote_folder: str, filenames_remote_folder: str, run: bool = False):
        """Save the list of files contained in `remote_folder` in a `.txt` file stored in `filenames_remote_folder`.
        Avoid using this while uploading files to the remote_folder.
        """
        self.to_execute.append((self._save_listdir, (Path(remote_folder), Path(filenames_remote_folder)), {}))
        if run: self.run()

    def run(self):
        """Start the connection to SFTP server and run the coroutines.
        """
        asyncio.run(self._exec())


def main():
    # Copy the dataset
    # Parametrize the SFTP connection
    share_link="kGhVq86ssd"
    input_path = "resized/valid/"
    output_path = "datasets/"

    connection = AsyncSFTP(sftp_params=AsyncSFTPParams(
            host="io.erda.au.dk",
            port=2222,
            username=share_link,
            password=share_link,
            ),
            log_file="sftp.log")

    
    # Copy the model
