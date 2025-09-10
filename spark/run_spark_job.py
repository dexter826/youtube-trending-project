"""
Spark Job Runner for Windows
Author: BigData Expert
Description: Run PySpark job with proper environment setup
"""

import os
import sys
import subprocess

def setup_environment():
    """Setup environment variables for Spark on Windows"""
    # Set Hadoop home
    os.environ['HADOOP_HOME'] = 'C:\\hadoop'
    
    # Add to PATH
    hadoop_bin = 'C:\\hadoop\\bin'
    if hadoop_bin not in os.environ.get('PATH', ''):
        os.environ['PATH'] = os.environ.get('PATH', '') + ';' + hadoop_bin
    
    # Set Java opts to increase memory and avoid warnings
    os.environ['SPARK_LOCAL_DIRS'] = 'C:\\temp\\spark'
    os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
    
    # Increase driver memory for large datasets
    os.environ['SPARK_DRIVER_MEMORY'] = '4g'
    os.environ['SPARK_EXECUTOR_MEMORY'] = '4g'
    
    # Create temp directory if not exists
    os.makedirs('C:\\temp\\spark', exist_ok=True)
    
    print("[OK] Environment variables set:")
    print(f"   HADOOP_HOME: {os.environ.get('HADOOP_HOME')}")
    print(f"   SPARK_DRIVER_MEMORY: {os.environ.get('SPARK_DRIVER_MEMORY')}")
    print(f"   PATH includes: {hadoop_bin}")

def run_spark_job(data_directory):
    """Run the Spark job"""
    setup_environment()
    
    # Path to the Spark job
    job_path = os.path.join(os.path.dirname(__file__), 'jobs', 'process_trending.py')
    
    print(f"🚀 Running Spark job: {job_path}")
    print(f"📁 Data directory: {data_directory}")
    
    # Run the job
    cmd = [sys.executable, job_path, data_directory]
    
    try:
        result = subprocess.run(cmd, capture_output=False, text=True)
        return result.returncode == 0
    except Exception as e:
        print(f"[ERROR] Error running Spark job: {e}")
        return False

def main():
    if len(sys.argv) != 2:
        print("Usage: python run_spark_job.py <data_directory>")
        print("Example: python run_spark_job.py ../data")
        sys.exit(1)
    
    data_directory = sys.argv[1]
    
    if not os.path.exists(data_directory):
        print(f"[ERROR] Data directory not found: {data_directory}")
        sys.exit(1)
    
    print("[STARTING] Spark Job for YouTube Trending Analysis")
    print("=" * 60)
    
    success = run_spark_job(data_directory)
    
    if success:
        print("=" * 60)
        print("[SUCCESS] Spark job completed successfully!")
    else:
        print("=" * 60)
        print("[FAILED] Spark job failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
