#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
#%% Imports

import os
import urllib.request
import urllib.error


# ---------------------------------------------------------------------------------------------------------------------
#%% Functions

def download_file_from_url(url, save_folder_path):
    
    '''
    Helper used to download a file and place it in a folder,
    with the file name being given based on the url
    '''
    
    # Build save path and make sure the save folder exists
    save_name = os.path.basename(url)
    save_path = os.path.join(save_folder_path, save_name)
    os.makedirs(save_folder_path, exist_ok=True)
    
    # If download fails, ask user to do it manually
    try:
        urllib.request.urlretrieve(url, save_path)
    except urllib.error.URLError as err:
        print("",
              "Error:",
              str(err),
              "",
              "Unable to download file!",
              "Please manually download the following file:",
              url,
              "",
              "And place it in the folder:",
              save_folder_path,
              "",
              sep="\n", flush=True)
        raise SystemExit()
    
    return save_path

def download_missing_model_files(urls_list, save_folder_path):
    
    '''
    Function used to download files from the given list of urls
    This will only run if the given folder path has no files in it!
    If a single file is in the folder already, then no downloads will occur
    '''
    
    # Bail if we already have files
    if os.path.exists(save_folder_path):
        already_have_files = len(os.listdir(save_folder_path)) > 0
        if already_have_files:
            return
        
    for url in urls_list:
        print("", "Downloading", f"@ {url}", sep = "\n", flush=True)
        save_path = download_file_from_url(url, save_folder_path)
        print("Saved:", save_path)
    
    return