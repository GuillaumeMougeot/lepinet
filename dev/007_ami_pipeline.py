import os
import asyncio
import asyncssh
from asyncssh import SFTPClient, SFTPError
from typing import TypedDict
from os.path import join
from tqdm.asyncio import tqdm as asynctqdm
from aiomultiprocess import Pool
from functools import partial
# from pyremotedata.implicit_mount import IOHandler
from tqdm import tqdm
import paramiko
import time
import subprocess

class AsyncSFTPParams(TypedDict):
    host: str
    port: int
    username: str
    password: str
    client_keys: list[str]

sftp_params=AsyncSFTPParams(
        host="io.erda.au.dk",
        port=2222,
        username="gmo@ecos.au.dk",
        client_keys=["~/.ssh/id_rsa"])

remotepath = "AMI/storage/mambo/2024/uva/NL1/2024_07_18"
localpath = "/home/george/codes/lepinet/data/mambo/images"

max_concurrent=16
semaphore = asyncio.Semaphore(max_concurrent)

async def get(sftp, remote_file, local_file, progress_bar, semaphore):
    async with semaphore:
        try:
            await sftp.get(remote_file, local_file)
        except SFTPError as e:
            print(f"Failed to download {remote_file}: {e}")
        finally:
            progress_bar.update(1)

async def main_v1():
    async with asyncssh.connect(**sftp_params) as conn:
        async with conn.start_sftp_client() as sftp:
            ls = await sftp.listdir(remotepath)
            # Select a subset (e.g., first 100 files)
            ls = ls[:100]
            download_progress_bar = asynctqdm(total=len(ls), desc="Downloading Images", unit="image", position=0)
            tasks = [
                get(sftp, join(remotepath, f), join(localpath, f), download_progress_bar, semaphore)
                for f in ls
            ]
            await asyncio.gather(*tasks)

def main_sync():
    transport = paramiko.Transport((sftp_params["host"], sftp_params["port"]))
    pkey = paramiko.RSAKey.from_private_key_file(os.path.expanduser(sftp_params["client_keys"][0]))
    transport.connect(username=sftp_params["username"], pkey=pkey)
    sftp = paramiko.SFTPClient.from_transport(transport)

    try:
        files = sftp.listdir(remotepath)
        files = files[:100]  # subset
        with tqdm(total=len(files), desc="Downloading Images", unit="image") as pbar:
            for f in files:
                remote_file = join(remotepath, f)
                local_file = join(localpath, f)
                try:
                    sftp.get(remote_file, local_file)
                except Exception as e:
                    print(f"Failed to download {remote_file}: {e}")
                pbar.update(1)
    finally:
        sftp.close()
        transport.close()

# async def download(p):
#     async with asyncssh.connect(**sftp_params) as conn:
#         async with conn.start_sftp_client() as sftp:
#             tasks = []
#             for rf, lf in p:
#                 tasks.append(await sftp.get(rf, lf))
#             await asyncio.gather(*tasks)

# async def main():
#     async with asyncssh.connect(**sftp_params) as conn:
#         async with conn.start_sftp_client() as sftp:
#             ls = await sftp.listdir(remotepath)

        
#     paths = [(join(remotepath, f), join(localpath, f)) for f in ls]
#     chunks = [paths[i::max_concurrent] for i in range(max_concurrent)]
            
#     with tqdm(total=len(chunks), desc="Downloading Files", unit="file") as progress_bar:
#         async with Pool(processes=max_concurrent) as pool:  # Adjust based on CPU cores
#             async for _ in  pool.map(download, chunks):
#                 progress_bar.update(1)

def run_sftp():
    try:
        asyncio.run(main_v1())
    except (OSError, asyncssh.Error) as exc:
        print(f'SFTP connection failed: {exc}')

# def download_lftp():
#     with IOHandler() as io:
#         print(io.ls())
        # io.download(remote_path=remotepath, local_destination=localpath, n=max_concurrent)

def benchmark_lftp_v1():
    remote_folder = "AMI/storage/mambo/2024/uva/NL1/2024_07_18"
    local_folder = "/home/george/codes/lepinet/data/mambo/images"

    cmd = [
        "lftp",
        "-e",
        f"mirror --verbose --continue --parallel=16 {remote_folder} {local_folder}; bye",
        "sftp://erda"
    ]

    print("Starting lftp benchmark...")
    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    end = time.time()
    duration = end - start

    print(f"lftp download time: {duration:.2f} seconds")

    if result.returncode != 0:
        print("lftp encountered an error:")
        print(result.stderr)
    else:
        print("lftp completed successfully.")


def benchmark_lftp():
    remote_dir = "AMI/storage/mambo/2024/uva/NL1/2024_07_18"
    local_dir = "/home/george/data/classif/mambo/images"

    # Step 1: get first 100 filenames from remote dir
    list_cmd = [
        "lftp",
        "-c",
        f"open sftp://erda; cd {remote_dir}; cls -1"
    ]

    print("Getting remote file list...")
    result = subprocess.run(list_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("Failed to list files:")
        print(result.stderr)
        return

    files = result.stdout.strip().splitlines()[:100]
    if not files:
        print("No files found.")
        return

    # Step 2: build lftp command to mget those files
    file_cmds = "\n".join([f'mget --continue --parallel=16 "{f}"' for f in files])
    lftp_script = f"""
    open sftp://erda
    cd {remote_dir}
    lcd {local_dir}
    set net:idle 10
    set net:timeout 30
    set mirror:parallel-transfer-count 16
    {file_cmds}
    bye
    """

    print("Starting lftp (first 100 files) benchmark...")
    start = time.time()
    download = subprocess.run(["lftp", "-c", lftp_script], capture_output=True, text=True)
    end = time.time()
    duration = end - start

    print(f"lftp (100 files) download time: {duration:.2f} seconds")

    if download.returncode != 0:
        print("lftp encountered an error:")
        print(download.stderr)
    else:
        print("lftp completed successfully.")

def benchmark():
    print("Starting sync benchmark...")
    start = time.time()
    # main_sync()
    end = time.time()
    print(f"Synchronous download time: {end - start:.2f} seconds")

    print("Starting async benchmark...")
    start = time.time()
    asyncio.run(main_v1())
    end = time.time()
    print(f"Asynchronous download time: {end - start:.2f} seconds")

if __name__=="__main__":
    os.makedirs(localpath, exist_ok=True)
    # run_sftp()
    # download_lftp()
    # benchmark()
    benchmark_lftp()