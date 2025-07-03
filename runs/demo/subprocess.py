import os
import glob
import subprocess

def run_command_with_conda(working_dir, conda_env, cmd):
    """
    Run a command in a specific working directory using a specific conda environment.
    Output goes directly to the terminal (so tqdm bars still render).
    """
    current_dir = os.getcwd()
    try:
        os.chdir(working_dir)
        # Build a proper argument list instead of a shell string
        full_cmd = ["conda", "run", "-n", conda_env,
                    "--no-capture-output",     # don’t buffer/capture output internally
                    ] + cmd
        
        print(f"Running: {' '.join(full_cmd)}")
        # Don't redirect anything—inherit the parent’s fds
        process = subprocess.Popen(full_cmd)
        ret = process.wait()
        if ret != 0:
            print(f"❌ Command failed with exit code {ret}")
            return False
        print("✅ Command succeeded")
        return True

    except Exception as e:
        print(f"Exception while running code: {e}")
        return False
    finally:
        os.chdir(current_dir)