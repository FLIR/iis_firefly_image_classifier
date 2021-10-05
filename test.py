if __name__=="__main__":

    import subprocess
    import os
    my_env = os.environ.copy()
    print(my_env)
    my_env["PYTHONHOME"] = "/opt/movidius/virtualenv-python/bin/python"
    my_env["PATH"] = "/opt/movidius/virtualenv-python/bin/python"
    command = "mvNCCompile -s 12 -o test_3  optimized.pb  -in=input -on=MobilenetV1/Predictions/Reshape_1"
    subprocess.run(.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=my_env)
