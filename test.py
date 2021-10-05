if __name__=="__main__":

    import subprocess
    import os
    # my_env = os.environ.copy()
    # print('##################################',my_env)
    # print('##################################',my_env["PATH"])
    # my_env["PYTHONHOME"] = "/opt/movidius/virtualenv-python/bin/python"
    # my_env["PATH"] = "/opt/movidius/virtualenv-python/bin/python"
    command = "/usr/local/bin/mvNCCompile -s 12 -o test_3  optimized.pb  -in=input -on=MobilenetV1/Predictions/Reshape_1"
    # subprocess.run(command.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=my_env)
    #os.system(command)
    os.environ['PYTHONPATH'] = "/opt/movidius/caffe/python" # visible in this process + all children
    subprocess.check_call(command.split(),
                      env=dict(os.environ, SQSUB_VAR="visible in this subprocess"))
