import yaml
import os
import re
import subprocess
import json
import hashlib
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ScriptError(Exception):
    """Custom exception class for script errors."""
    pass

def read_yaml_config(file_path):
    with open(file_path, 'r') as file:
        config = yaml.full_load(file)
        return config['property']['model-engine-file']

def extract_model_version(model_dir_configmap):
    match = re.search(r'(?<=/output/deepstream_models/).*(?=/)', model_dir_configmap)
    if match:
        return match.group(0)
    else:
        raise ScriptError("Error extracting model version from path.")

def get_auth_credentials(dockerconfigjson, jfrog_env):
    auths = json.loads(dockerconfigjson)
    username = auths["auths"][jfrog_env]["username"]
    password = auths["auths"][jfrog_env]["password"]
    if not username or not password:
        raise ScriptError('Failed to retrieve username or password')
    return username, password

def get_gpu_model():
    try:
        gpu_info = subprocess.check_output(['nvidia-smi', '-L']).decode().strip()
        match = re.search(r'(?<=: ).*', gpu_info)
        return match.group(0) if match else None
    except subprocess.CalledProcessError as e:
        raise ScriptError(f"Error retrieving GPU model: {e}")

def run_curl_command(url, username, password, local_path):
    curl_command = [
    'curl', '--connect-timeout', '30', '--retry', '3', '--retry-delay', '5',
    '--user', f'{username}:{password}', '-s', '-S', '-f', '-O', url]
    try:
        subprocess.run(curl_command, check=True)
        os.rename(os.path.basename(url), local_path)
        logging.info(f"Downloaded file to {local_path}")
    except Exception as e:
        raise ScriptError(f"Error downloading file: {e}")

def get_jfrog_checksum(url, username, password):
    curl_command = [
    'curl', '--user', f'{username}:{password}', '-s', url]
    try:
        result = subprocess.check_output(curl_command).decode()
        response_json = json.loads(result)
        return response_json['checksums']['md5']
    except Exception as e:
        raise ScriptError(f"Error fetching JFrog checksum: {e}")

def verify_md5(local_file_path, expected_md5):
    hash_md5 = hashlib.md5()
    with open(local_file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    md5_matches = hash_md5.hexdigest() == expected_md5
    if not md5_matches:
        raise ScriptError(f"MD5 check failed for {local_file_path}.")
    logging.info(f"MD5 check passed for {local_file_path}")

def download_and_check_label_file(label_file, model_version, username, password):
    if not os.path.isfile(label_file):
        logging.info("Label file not found. Downloading label file.")
        label_url = f"https://colesgroup.jfrog.io/artifactory/ieb-prod-generic-virtual/deepstream-models/{model_version}/labels.txt"
        run_curl_command(label_url, username, password, label_file)
    else:
        logging.info("Label file already exists. Skipping download.")

def download_and_check_model_file(model_file, model_version, gpu_model, username, password):
    
    if not os.path.isfile(model_file):
        logging.info("Model file not found. Downloading model file for GPU model...")

        if 'A2' in gpu_model:
            model_url = f"https://colesgroup.jfrog.io/artifactory/ieb-prod-generic-virtual/deepstream-models/{model_version}/A2_model.onnx_b1_gpu0_fp16.engine"
            local_path = 'A2_model.onnx_b1_gpu0_fp16.engine'

        elif 'A16' in gpu_model:
            model_url = f"https://colesgroup.jfrog.io/artifactory/ieb-prod-generic-virtual/deepstream-models/{model_version}/A16_model.onnx_b1_gpu0_fp16.engine"
            local_path = 'A16_model.onnx_b1_gpu0_fp16.engine'

        else:
            raise ScriptError(f"Unsupported GPU model {gpu_model}")

        run_curl_command(model_url, username, password, local_path)

        jfrog_url = f"https://colesgroup.jfrog.io/artifactory/api/storage/ieb-prod-generic-virtual/deepstream-models/{model_version}/{os.path.basename(local_path)}"
        jfrog_checksum = get_jfrog_checksum(jfrog_url, username, password)
        verify_md5(local_path, jfrog_checksum)
        os.rename(local_path, model_file)
        logging.info(f"Downloaded and verified model file - {os.stat(model_file)}")
    else:
        logging.info("Model file already exists. Skipping download.")
        

def ensure_model_directory_exists(model_dir):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        logging.info(f"Created model directory: {model_dir}")
    else:
        logging.info(f"Model directory already exists: {model_dir}")
    
    return model_dir

def main():

    try:

        model_dir_configmap = read_yaml_config('/config/config-infer-primary-bot.yaml')

        model_version = extract_model_version(model_dir_configmap)
        
        logging.info(f"Model version: {model_version}")

        dockerconfigjson = os.getenv('dockerconfigjson')

        jfrog_env = os.getenv('JFROG_ENV')

        if not dockerconfigjson or not jfrog_env:
            raise ScriptError("Docker config or JFrog environment variable not set.")

        username, password = get_auth_credentials(dockerconfigjson, jfrog_env)
        
        os.chdir('/output')
        
        model_dir = f"/output/deepstream_models/{model_version}"
        
        ensure_model_directory_exists(model_dir)

        label_file = f"/output/deepstream_models/{model_version}/labels.txt"
        download_and_check_label_file(label_file, model_version, username, password)

        gpu_model = get_gpu_model()

        model_dir = f"/output/deepstream_models/{model_version}"
        model_file = f"{model_dir}/model.onnx_b1_gpu0_fp16.engine"
        download_and_check_model_file(model_file, model_version, gpu_model, username, password)

    except ScriptError as e:
        logging.error(e)
        exit(1)
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        exit(1)

if __name__ == '__main__':
    main()